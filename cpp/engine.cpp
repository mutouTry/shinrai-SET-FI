// setfi._engine: event-driven SET pulse propagation to the flip-flop D pins.
//
// Three entry points:
//   simulate_pulses   inject one SET per pulse width on one source net and
//                     propagate it through a packed fan-out cone; returns the
//                     event stream seen at every recorded FF D net.
//   extract_pulses    turn a recorded event stream into pulse intervals
//                     [t_enter, t_exit) relative to the SET start.
//   run_block         every (cycle, site, pulse width) trial of a block of
//                     cycles: simulate_pulses and extract_pulses on compiled
//                     cones, on several threads.
//
// The propagation model (see docs/MODEL.md):
//   * logic: every gate is re-evaluated from its truth table on each input
//     change (logical masking);
//   * delay: an output transition takes the SDF IOPATH delay of an input that
//     changed, for the rise or fall at the output, with the COND entry that
//     holds for the side-input values after the change.  Every changed input
//     whose path is active competes and the smallest delay is used (Verilog
//     module-path delays).  A path is active unless its arc has COND entries,
//     the side-input values are known and no COND entry for this output edge
//     holds.  If no changed input has an active path, the changed inputs that
//     alone explain the output change (all changed inputs if none does)
//     compete with their default delays, and the max (or min) is used;
//   * inertial delay: one pending transition per gate output; it is cancelled
//     when the gate, re-evaluated on an input change before the transition
//     fires, gives the current output value again, so a pulse narrower than
//     the gate delay dies there;
//   * optional transport wires (SDF INTERCONNECT) delay but never filter;
//   * vcs_align (propagation.vcs_align): an arc with an unconditional SDF entry
//     is active with that delay when none of its COND entries holds, and when no
//     changed input has an active path but one has an inactive one, the output
//     changes with zero delay, as VCS does for state-dependent paths.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <exception>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "fastpath.hpp"

namespace py = pybind11;

namespace {

struct SimStats {
  int64_t applied_events = 0;
  int64_t canceled_replace = 0;
  int64_t canceled_back_to_current = 0;
  int64_t kept_same_target = 0;
  int64_t max_heap = 0;
  int64_t coalesced_nets = 0;      // nets hit by >1 event in one time batch
  int64_t coalesced_events = 0;    // events dropped by last-token-wins coalescing
  int64_t dropped_arc_no_delay = 0;
  int64_t dropped_no_candidate_arc = 0;
};

using I32Arr = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;
using I64Arr = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;
using U8Arr = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;
using I8Arr = py::array_t<int8_t, py::array::c_style | py::array::forcecast>;

template <typename T>
static py::array_t<T, py::array::c_style | py::array::forcecast>
as_1d_array(const py::object& obj, const char* name) {
  try {
    return py::cast<py::array_t<T, py::array::c_style | py::array::forcecast>>(obj);
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("cannot convert array '") + name + "': " + e.what());
  }
}

template <typename T>
struct ArrView {
  const T* p = nullptr;
  int64_t n = 0;
};

template <typename T>
static ArrView<T> view_1d(const py::array_t<T, py::array::c_style | py::array::forcecast>& a,
                          const char* name) {
  auto info = a.request();
  if (info.ndim != 1) throw std::runtime_error(std::string("array '") + name + "' must be 1-D");
  ArrView<T> v;
  v.p = static_cast<const T*>(info.ptr);
  v.n = static_cast<int64_t>(info.shape[0]);
  return v;
}

// Arc delay lookup.  Unconditional rise/fall delay per arc, optionally
// overridden by the first matching conditional pattern (side-input values).
// A pattern delay < 0 means "use the unconditional delay".
// active_only: return -1 instead of the unconditional delay when the arc has
// conditional patterns and none for this output edge matches: its path is
// inactive (a Verilog state-dependent path whose condition is false).  An arc
// without patterns, or whose side-input values are unknown, is always active.
static inline int64_t delay_get_ps(const ArrView<int64_t>& default_rise_ps,
                                   const ArrView<int64_t>& default_fall_ps,
                                   const ArrView<int32_t>& patt_ptr,
                                   const ArrView<int64_t>& patt_mask,
                                   const ArrView<int64_t>& patt_valbits,
                                   const ArrView<int64_t>& patt_dt_ps,
                                   const ArrView<uint8_t>& patt_is_rise,
                                   int32_t dkey_id, int tr_is_rise, bool sbits_valid,
                                   int64_t sbits, bool active_only) {
  if (dkey_id < 0) return 0;
  tr_is_rise = tr_is_rise ? 1 : 0;
  if (dkey_id >= default_rise_ps.n || dkey_id >= default_fall_ps.n) return 0;

  const int64_t def_ps = tr_is_rise ? default_rise_ps.p[dkey_id] : default_fall_ps.p[dkey_id];
  if (!sbits_valid) return def_ps;
  if (dkey_id + 1 >= patt_ptr.n) return def_ps;
  const int32_t a = patt_ptr.p[dkey_id];
  const int32_t b = patt_ptr.p[dkey_id + 1];
  if (a >= b) return def_ps;

  for (int32_t i = a; i < b; i++) {
    if (i < 0 || i >= patt_is_rise.n) break;
    if ((int)patt_is_rise.p[i] != tr_is_rise) continue;
    if ((sbits & patt_mask.p[i]) == patt_valbits.p[i]) {
      const int64_t dt = patt_dt_ps.p[i];
      return (dt < 0) ? def_ps : dt;
    }
  }
  return active_only ? -1 : def_ps;
}

struct Event {
  int64_t t_ps;
  int32_t tok;
  int32_t net_i;
  uint8_t val;
};

// min-heap by (t_ps, tok): same-time events are processed in push order
struct EventCmp {
  bool operator()(const Event& a, const Event& b) const {
    if (a.t_ps != b.t_ps) return a.t_ps > b.t_ps;
    return a.tok > b.tok;
  }
};

static py::dict stats_dict(const SimStats& s) {
  py::dict d;
  d["applied_events"] = s.applied_events;
  d["canceled_replace"] = s.canceled_replace;
  d["canceled_back_to_current"] = s.canceled_back_to_current;
  d["kept_same_target"] = s.kept_same_target;
  d["max_heap"] = s.max_heap;
  d["coalesced_nets"] = s.coalesced_nets;
  d["coalesced_events"] = s.coalesced_events;
  d["dropped_arc_no_delay"] = s.dropped_arc_no_delay;
  d["dropped_no_candidate_arc"] = s.dropped_no_candidate_arc;
  return d;
}

// ---------------------------------------------------------------------------
// simulate_pulses
// ---------------------------------------------------------------------------
// Read-only views of one packed cone (setfi.campaign.cone.PackedCone) and of the
// delay table (setfi.campaign.substrate.DelayTable).
struct ConeView {
  ArrView<int32_t> fanout_ptr, fanout_idx, gate_out, gate_var_ptr, gate_var_nets, gate_truth_ptr;
  ArrView<uint8_t> gate_truth;
  ArrView<int32_t> arc_dkey_id, arc_cause_i;
  ArrView<int8_t> arc_side_mode;
  ArrView<int32_t> arc_side_ptr, arc_side_nets, rec_idx;
  // Optional transport wires (SDF INTERCONNECT), CSR over driver nets, sorted
  // zero-delay first: [wire_ptr[n], wire_zsplit[n]) have d == 0 and are
  // propagated inside the current time batch; [wire_zsplit[n], wire_ptr[n+1])
  // have d > 0 and become one heap event per edge.  Read only when has_wires.
  ArrView<int32_t> wire_ptr, wire_zsplit, wire_sink;
  ArrView<int64_t> wire_rise_ps, wire_fall_ps;
  bool has_wires = false;
};

struct DelayView {
  ArrView<int64_t> default_rise, default_fall;
  ArrView<int32_t> patt_ptr;
  ArrView<int64_t> patt_mask, patt_valbits, patt_dt_ps;
  ArrView<uint8_t> patt_is_rise;
  ArrView<uint8_t> uncond_rise, uncond_fall;  // per arc: an unconditional SDF entry (may be empty)
};

// per record slot: the (time, new value) of every transition applied to it
using RecEvents = std::vector<std::vector<std::pair<int64_t, uint8_t>>>;

// The propagation kernel of simulate_pulses, free of Python objects so that
// run_block can call it without the GIL.  For every pulse width it fills
// rec_events (one list per record slot) and then calls on_result(pw_idx, stats).
// The propagation flags are the trailing arguments of simulate_pulses, in the
// same order; run_block passes them from its EngineOptions.
template <typename OnResult>
static void simulate_pulses_core(const ConeView& cv, const DelayView& dv, const uint8_t* init_p,
                                 int32_t src_i, int32_t v0, int32_t v1,
                                 const int64_t* pw_ps_p, int64_t n_pw, int64_t sim_end_ps, int64_t eps_ps,
                                 bool delay_policy_is_max, bool vcs_align, RecEvents& rec_events,
                                 OnResult&& on_result) {
  const ArrView<int32_t>& fanout_ptr = cv.fanout_ptr;
  const ArrView<int32_t>& fanout_idx = cv.fanout_idx;
  const ArrView<int32_t>& gate_out = cv.gate_out;
  const ArrView<int32_t>& gate_var_ptr = cv.gate_var_ptr;
  const ArrView<int32_t>& gate_var_nets = cv.gate_var_nets;
  const ArrView<int32_t>& gate_truth_ptr = cv.gate_truth_ptr;
  const ArrView<uint8_t>& gate_truth = cv.gate_truth;
  const ArrView<int32_t>& arc_dkey_id = cv.arc_dkey_id;
  const ArrView<int32_t>& arc_cause_i = cv.arc_cause_i;
  const ArrView<int8_t>& arc_side_mode = cv.arc_side_mode;
  const ArrView<int32_t>& arc_side_ptr = cv.arc_side_ptr;
  const ArrView<int32_t>& arc_side_nets = cv.arc_side_nets;
  const ArrView<int32_t>& rec_idx = cv.rec_idx;
  const ArrView<int32_t>& wire_ptr = cv.wire_ptr;
  const ArrView<int32_t>& wire_zsplit = cv.wire_zsplit;
  const ArrView<int32_t>& wire_sink = cv.wire_sink;
  const ArrView<int64_t>& wire_rise_ps = cv.wire_rise_ps;
  const ArrView<int64_t>& wire_fall_ps = cv.wire_fall_ps;
  const bool has_wires = cv.has_wires;
  const ArrView<int64_t>& default_rise = dv.default_rise;
  const ArrView<int64_t>& default_fall = dv.default_fall;
  const ArrView<int32_t>& patt_ptr = dv.patt_ptr;
  const ArrView<int64_t>& patt_mask = dv.patt_mask;
  const ArrView<int64_t>& patt_valbits = dv.patt_valbits;
  const ArrView<int64_t>& patt_dt_ps = dv.patt_dt_ps;
  const ArrView<uint8_t>& patt_is_rise = dv.patt_is_rise;
  auto has_uncond = [&](int32_t dkey_id, int tr_is_rise) -> bool {
    const ArrView<uint8_t>& u = tr_is_rise ? dv.uncond_rise : dv.uncond_fall;
    return dkey_id >= 0 && dkey_id < u.n && u.p[dkey_id] != 0;
  };

  const int32_t n_nets = (int32_t)rec_idx.n;
  const int32_t n_gates = (int32_t)gate_out.n;

  int32_t n_rec_slots = 0;
  for (int32_t i = 0; i < n_nets; i++) {
    const int32_t r = rec_idx.p[i];
    if (r >= 0) n_rec_slots = std::max(n_rec_slots, (int32_t)(r + 1));
  }
  rec_events.resize((size_t)n_rec_slots);

  for (int64_t pw_idx = 0; pw_idx < n_pw; ++pw_idx) {
    const int64_t pw_ps = pw_ps_p[pw_idx];
    SimStats stats;

    if (src_i < 0) {
      for (auto& lst : rec_events) lst.clear();
      on_result(pw_idx, stats);
      continue;
    }

    for (auto& lst : rec_events) lst.clear();

    std::vector<uint8_t> net_val((size_t)n_nets, 0);
    std::copy(init_p, init_p + n_nets, net_val.begin());
    for (int32_t i = 0; i < n_nets; i++) net_val[i] &= 1;

    std::vector<int32_t> old_tag((size_t)n_nets, 0);
    std::vector<uint8_t> old_val((size_t)n_nets, 0);
    int32_t tag = 1;

    std::vector<int32_t> gate_mark((size_t)n_gates, 0);
    int32_t gate_mark_tag = 1;

    // token bookkeeping, indexed by token
    std::vector<uint8_t> canceled(2, 0);
    std::vector<int32_t> tok2outnet(2, -1);
    std::vector<uint8_t> tok2pend_val(2, 0);

    auto ensure_tok = [&](int32_t tok) {
      if (tok >= (int32_t)canceled.size()) {
        const size_t new_sz = (size_t)tok + 1;
        canceled.resize(new_sz, 0);
        tok2outnet.resize(new_sz, -1);
        tok2pend_val.resize(new_sz, 0);
      }
    };

    std::vector<int32_t> pending_tok_by_outnet((size_t)n_nets, -1);

    std::priority_queue<Event, std::vector<Event>, EventCmp> heap;
    int32_t token_ctr = 1;

    auto prev_val = [&](int32_t ni) -> int32_t {
      if (old_tag[(size_t)ni] == tag) return (int32_t)(old_val[(size_t)ni] & 1);
      return (int32_t)(net_val[(size_t)ni] & 1);
    };

    auto push_event = [&](int64_t t_ps, int32_t net_i, int32_t val) -> int32_t {
      const int32_t tok = token_ctr++;
      ensure_tok(tok);
      Event e;
      e.t_ps = t_ps;
      e.tok = tok;
      e.net_i = net_i;
      e.val = (uint8_t)(val & 1);
      heap.push(e);
      if ((int64_t)heap.size() > stats.max_heap) stats.max_heap = (int64_t)heap.size();
      return tok;
    };

    // cancel a pending transition: its event is skipped when it comes off the heap
    enum CancelWhy { WHY_REPLACE, WHY_BACK };
    auto cancel_token = [&](int32_t tok, CancelWhy why) {
      if (tok <= 0) return;
      ensure_tok(tok);
      if (canceled[(size_t)tok]) return;
      canceled[(size_t)tok] = 1;
      const int32_t outn = tok2outnet[(size_t)tok];
      if (outn >= 0) {
        if (pending_tok_by_outnet[(size_t)outn] == tok) pending_tok_by_outnet[(size_t)outn] = -1;
        tok2outnet[(size_t)tok] = -1;
      }
      tok2pend_val[(size_t)tok] = 0;
      if (why == WHY_REPLACE) stats.canceled_replace++;
      else stats.canceled_back_to_current++;
    };

    // the SET: flip the source net at t=0, restore it at t=pw
    push_event(0, src_i, v1);
    push_event(pw_ps, src_i, v0);

    std::vector<int32_t> best_tok((size_t)n_nets, -1);
    std::vector<uint8_t> best_val((size_t)n_nets, 0);
    std::vector<int32_t> touched;
    touched.reserve(1024);

    // A wire delays a pulse and never masks one: these helpers never touch the
    // pending-transition bookkeeping, so a wire cannot cancel anything.
    auto propagate_zero_wires = [&]() {
      if (!has_wires) return;
      const size_t n0 = touched.size();
      for (size_t ti = 0; ti < n0; ti++) {
        const int32_t cn = touched[ti];
        if (cn < 0 || cn >= n_nets) continue;
        for (int32_t w = wire_ptr.p[cn]; w < wire_zsplit.p[cn]; w++) {
          const int32_t sk = wire_sink.p[w];
          if (sk < 0 || sk >= n_nets) continue;
          if (best_tok[(size_t)sk] < 0) touched.push_back(sk);
          best_tok[(size_t)sk] = 0;  // not a real token
          best_val[(size_t)sk] = (uint8_t)(best_val[(size_t)cn] & 1);
        }
      }
    };
    auto propagate_delayed_wires = [&](const std::vector<int32_t>& chg, int64_t t_now) {
      if (!has_wires) return;
      for (int32_t cn : chg) {
        if (cn < 0 || cn >= n_nets) continue;
        const int32_t wa = wire_zsplit.p[cn], wb = wire_ptr.p[cn + 1];
        if (wa >= wb) continue;
        const uint8_t v = net_val[(size_t)cn] & 1;
        for (int32_t w = wa; w < wb; w++) {
          const int32_t sk = wire_sink.p[w];
          if (sk < 0 || sk >= n_nets) continue;
          push_event(t_now + (v ? wire_rise_ps.p[w] : wire_fall_ps.p[w]), sk, (int32_t)v);
        }
      }
    };

    while (!heap.empty()) {
      const Event e0 = heap.top();
      heap.pop();
      const int64_t t0_ps = e0.t_ps;
      if (t0_ps > sim_end_ps) break;

      std::vector<Event> batch;
      batch.reserve(256);
      batch.push_back(e0);
      if (eps_ps == 0) {
        while (!heap.empty() && heap.top().t_ps == t0_ps) {
          batch.push_back(heap.top());
          heap.pop();
        }
      } else {
        while (!heap.empty() && (heap.top().t_ps - t0_ps) <= eps_ps) {
          batch.push_back(heap.top());
          heap.pop();
        }
      }

      if (batch.size() >= 2) {
        std::unordered_map<int32_t, int32_t> seen_count;
        seen_count.reserve(batch.size() * 2);
        for (const auto& ev : batch) seen_count[ev.net_i]++;
        for (const auto& kv : seen_count) {
          if (kv.second > 1) {
            stats.coalesced_nets++;
            stats.coalesced_events += (kv.second - 1);
          }
        }
      }

      for (int32_t ni : touched) best_tok[(size_t)ni] = -1;
      touched.clear();

      // last token wins per net within one batch
      for (const auto& ev : batch) {
        const int32_t tok = ev.tok;
        ensure_tok(tok);
        if (canceled[(size_t)tok]) continue;
        const int32_t ni = ev.net_i;
        const int32_t bt = best_tok[(size_t)ni];
        if (bt < tok) {
          if (bt < 0) touched.push_back(ni);
          best_tok[(size_t)ni] = tok;
          best_val[(size_t)ni] = (uint8_t)(ev.val & 1);
        }
      }
      propagate_zero_wires();
      if (touched.empty()) continue;

      tag++;
      if (tag >= 1'000'000'000) {
        std::fill(old_tag.begin(), old_tag.end(), 0);
        tag = 1;
      }

      std::vector<int32_t> changed_nets;
      changed_nets.reserve(touched.size());
      for (int32_t ni : touched) {
        const uint8_t vv = best_val[(size_t)ni] & 1;
        const uint8_t cur = net_val[(size_t)ni] & 1;
        if (vv == cur) continue;
        old_tag[(size_t)ni] = tag;
        old_val[(size_t)ni] = cur;
        net_val[(size_t)ni] = vv;
        changed_nets.push_back(ni);
        stats.applied_events++;

        const int32_t rpos = rec_idx.p[ni];
        if (rpos >= 0 && rpos < n_rec_slots) rec_events[(size_t)rpos].push_back({t0_ps, vv});

        const int32_t tok = best_tok[(size_t)ni];
        if (tok > 0) {
          ensure_tok(tok);
          const int32_t outn = tok2outnet[(size_t)tok];
          if (outn >= 0 && pending_tok_by_outnet[(size_t)outn] == tok) {
            pending_tok_by_outnet[(size_t)outn] = -1;
            tok2outnet[(size_t)tok] = -1;
            tok2pend_val[(size_t)tok] = 0;
          }
        }
      }
      if (changed_nets.empty()) continue;

      propagate_delayed_wires(changed_nets, t0_ps);

      gate_mark_tag++;
      if (gate_mark_tag >= 1'000'000'000) {
        std::fill(gate_mark.begin(), gate_mark.end(), 0);
        gate_mark_tag = 1;
      }

      std::vector<int32_t> affected;
      affected.reserve(1024);
      for (int32_t cn : changed_nets) {
        for (int32_t k = fanout_ptr.p[cn]; k < fanout_ptr.p[cn + 1]; k++) {
          const int32_t gid = fanout_idx.p[k];
          if (gid < 0 || gid >= n_gates) continue;
          if (gate_mark[(size_t)gid] != gate_mark_tag) {
            gate_mark[(size_t)gid] = gate_mark_tag;
            affected.push_back(gid);
          }
        }
      }
      if (affected.empty()) continue;

      for (int32_t gid : affected) {
        const int32_t out_i = gate_out.p[gid];
        const int32_t va = gate_var_ptr.p[gid];
        const int32_t vb = gate_var_ptr.p[gid + 1];
        const int32_t k_in = vb - va;

        int32_t idx_prev = 0;
        int32_t idx_cur = 0;
        for (int32_t j = 0; j < k_in; j++) {
          const int32_t ni = gate_var_nets.p[va + j];
          idx_prev |= ((prev_val(ni) & 1) << j);
          idx_cur |= (((int32_t)net_val[(size_t)ni] & 1) << j);
        }
        const int32_t changed_mask = idx_prev ^ idx_cur;
        if (changed_mask == 0) continue;

        const uint8_t y_cur = net_val[(size_t)out_i] & 1;
        const int32_t ta = gate_truth_ptr.p[gid];
        const int32_t tb = gate_truth_ptr.p[gid + 1];
        const uint8_t y_new = (tb - ta <= 1) ? (gate_truth.p[ta] & 1) : (gate_truth.p[ta + idx_cur] & 1);

        // Inertial delay: the gate gives its current value again before the
        // pending transition fires, so the pending transition is cancelled.
        if (y_new == y_cur) {
          const int32_t old_tok = pending_tok_by_outnet[(size_t)out_i];
          if (old_tok > 0) {
            ensure_tok(old_tok);
            if (!canceled[(size_t)old_tok] && tok2pend_val[(size_t)old_tok] != y_cur)
              cancel_token(old_tok, WHY_BACK);
          }
          continue;
        }

        const int tr_is_rise = (y_cur == 0 && y_new == 1) ? 1 : 0;

        // Delay of the arc from input ppos for this output transition, its COND
        // patterns evaluated on the side inputs' current (post-change) values; with
        // active_only, -1 marks an inactive path.  false: the input has no delay
        // arc and cannot carry the transition.
        auto arc_delay = [&](int32_t ppos, bool active_only, int64_t& d_used) -> bool {
          const int32_t idx = va + ppos;
          const int32_t dkey_id = arc_dkey_id.p[idx];
          if (dkey_id < 0 || arc_cause_i.p[idx] < 0) return false;
          const int8_t mode = arc_side_mode.p[idx];
          bool sbits_valid = true;
          int64_t sbits = 0;
          if (mode == 0) {
            sbits_valid = false;
          } else if (mode == 1) {
            sbits = 0;
          } else {
            const int32_t sa = arc_side_ptr.p[idx];
            const int32_t sb = arc_side_ptr.p[idx + 1];
            int64_t sbv = 0;
            bool missing = false;
            for (int32_t ib = sa; ib < sb; ib++) {
              const int32_t sni = arc_side_nets.p[ib];
              if (sni < 0) {
                missing = true;
                break;
              }
              if (net_val[(size_t)sni] & 1) sbv |= (1LL << (ib - sa));
            }
            if (missing) sbits_valid = false;
            else sbits = sbv;
          }
          d_used = delay_get_ps(default_rise, default_fall, patt_ptr, patt_mask, patt_valbits,
                                patt_dt_ps, patt_is_rise, dkey_id, tr_is_rise, sbits_valid, sbits,
                                active_only);
          if (d_used < 0 && vcs_align && has_uncond(dkey_id, tr_is_rise))
            d_used = tr_is_rise ? default_rise.p[dkey_id] : default_fall.p[dkey_id];
          return true;
        };

        // Verilog module-path delays: every changed input whose path is active
        // competes, and the smallest delay is used.
        bool found = false, inactive = false;
        int64_t best_delay = 0;
        for (int32_t j = 0; j < k_in; j++) {
          if (!((changed_mask >> j) & 1)) continue;
          int64_t d_used = 0;
          if (!arc_delay(j, true, d_used)) continue;   // no arc
          if (d_used < 0) {                            // inactive
            inactive = true;
            continue;
          }
          if (!found || d_used < best_delay) {
            found = true;
            best_delay = d_used;
          }
        }
        if (!found && inactive && vcs_align) found = true;   // best_delay = 0

        // No changed input has an active path: the changed inputs that alone
        // explain the output change (all changed inputs if none does) compete
        // with their default delays, and delay_choice picks the max or the min.
        if (!found) {
          std::vector<int32_t> cands;
          cands.reserve(8);
          {
            int32_t m = changed_mask;
            int32_t ppos = 0;
            while (m) {
              if (m & 1) {
                const int32_t idx_ovr = (idx_prev & ~(1 << ppos)) | (idx_cur & (1 << ppos));
                const uint8_t yy = (tb - ta <= 1) ? (gate_truth.p[ta] & 1) : (gate_truth.p[ta + idx_ovr] & 1);
                if (yy != y_cur) cands.push_back(ppos);
              }
              m >>= 1;
              ppos++;
            }
          }
          if (cands.empty()) {
            for (int32_t j = 0; j < k_in; j++)
              if ((changed_mask >> j) & 1) cands.push_back(j);
          }
          int64_t best_d = delay_policy_is_max ? -1 : (std::numeric_limits<int64_t>::max)();
          for (int32_t pp : cands) {
            int64_t d_used = 0;
            if (!arc_delay(pp, false, d_used)) {
              stats.dropped_arc_no_delay++;
              continue;
            }
            if (delay_policy_is_max ? (d_used > best_d) : (d_used < best_d)) {
              best_d = d_used;
              best_delay = d_used;
              found = true;
            }
          }
        }
        if (!found) {
          stats.dropped_no_candidate_arc++;
          continue;
        }

        const int32_t old_tok = pending_tok_by_outnet[(size_t)out_i];
        if (old_tok > 0) {
          ensure_tok(old_tok);
          if (!canceled[(size_t)old_tok]) {
            if (tok2pend_val[(size_t)old_tok] == (y_new & 1)) {
              stats.kept_same_target++;
              continue;
            }
            cancel_token(old_tok, WHY_REPLACE);
          }
        }

        const int64_t t_out = t0_ps + best_delay;
        const int32_t tok_new = push_event(t_out, out_i, (int32_t)(y_new & 1));
        pending_tok_by_outnet[(size_t)out_i] = tok_new;
        ensure_tok(tok_new);
        tok2outnet[(size_t)tok_new] = out_i;
        tok2pend_val[(size_t)tok_new] = (uint8_t)(y_new & 1);
      }
    }

    on_result(pw_idx, stats);
  }
}

static ConeView cone_view_of_packed(const py::object& packed_obj, std::vector<py::object>& keep) {
  ConeView cv;
  auto i32 = [&](const char* name, ArrView<int32_t>& v) {
    I32Arr a = as_1d_array<int32_t>(packed_obj.attr(name), name);
    v = view_1d(a, name);
    keep.push_back(a);
  };
  i32("fanout_ptr", cv.fanout_ptr);
  i32("fanout_idx", cv.fanout_idx);
  i32("gate_out", cv.gate_out);
  i32("gate_var_ptr", cv.gate_var_ptr);
  i32("gate_var_nets", cv.gate_var_nets);
  i32("gate_truth_ptr", cv.gate_truth_ptr);
  {
    U8Arr a = as_1d_array<uint8_t>(packed_obj.attr("gate_truth"), "gate_truth");
    cv.gate_truth = view_1d(a, "gate_truth");
    keep.push_back(a);
  }
  i32("arc_dkey_id", cv.arc_dkey_id);
  i32("arc_cause_i", cv.arc_cause_i);
  {
    I8Arr a = as_1d_array<int8_t>(packed_obj.attr("arc_side_mode"), "arc_side_mode");
    cv.arc_side_mode = view_1d(a, "arc_side_mode");
    keep.push_back(a);
  }
  i32("arc_side_ptr", cv.arc_side_ptr);
  i32("arc_side_nets", cv.arc_side_nets);
  i32("rec_idx", cv.rec_idx);
  if (py::hasattr(packed_obj, "wire_sink")) {
    i32("wire_ptr", cv.wire_ptr);
    i32("wire_zsplit", cv.wire_zsplit);
    i32("wire_sink", cv.wire_sink);
    I64Arr r = as_1d_array<int64_t>(packed_obj.attr("wire_rise_ps"), "wire_rise_ps");
    I64Arr f = as_1d_array<int64_t>(packed_obj.attr("wire_fall_ps"), "wire_fall_ps");
    cv.wire_rise_ps = view_1d(r, "wire_rise_ps");
    cv.wire_fall_ps = view_1d(f, "wire_fall_ps");
    keep.push_back(r);
    keep.push_back(f);
    if (cv.wire_ptr.n != cv.rec_idx.n + 1 || cv.wire_zsplit.n != cv.rec_idx.n)
      throw std::runtime_error("wire_ptr/wire_zsplit length mismatch with n_nets");
    if (cv.wire_rise_ps.n != cv.wire_sink.n || cv.wire_fall_ps.n != cv.wire_sink.n)
      throw std::runtime_error("wire_sink/wire_rise_ps/wire_fall_ps length mismatch");
    cv.has_wires = (cv.wire_sink.n > 0);
  }
  return cv;
}

static DelayView delay_view_of(const py::object& delaytab_obj, std::vector<py::object>& keep) {
  DelayView dv;
  auto i64 = [&](const char* name, ArrView<int64_t>& v) {
    I64Arr a = as_1d_array<int64_t>(delaytab_obj.attr(name), name);
    v = view_1d(a, name);
    keep.push_back(a);
  };
  i64("default_rise_ps", dv.default_rise);
  i64("default_fall_ps", dv.default_fall);
  {
    I32Arr a = as_1d_array<int32_t>(delaytab_obj.attr("patt_ptr"), "patt_ptr");
    dv.patt_ptr = view_1d(a, "patt_ptr");
    keep.push_back(a);
  }
  i64("patt_mask", dv.patt_mask);
  i64("patt_valbits", dv.patt_valbits);
  i64("patt_dt_ps", dv.patt_dt_ps);
  {
    U8Arr a = as_1d_array<uint8_t>(delaytab_obj.attr("patt_is_rise"), "patt_is_rise");
    dv.patt_is_rise = view_1d(a, "patt_is_rise");
    keep.push_back(a);
  }
  for (auto [name, v] : {std::make_pair("uncond_rise", &dv.uncond_rise), std::make_pair("uncond_fall", &dv.uncond_fall)}) {
    if (!py::hasattr(delaytab_obj, name) || delaytab_obj.attr(name).is_none()) continue;
    U8Arr a = as_1d_array<uint8_t>(delaytab_obj.attr(name), name);
    *v = view_1d(a, name);
    keep.push_back(a);
  }
  return dv;
}

// Python entry point for one site and one bound state: one (ev_ptr, ev_t, ev_v,
// stats) tuple per pulse width.
static py::list simulate_pulses(py::object packed_obj, py::object delaytab_obj,
                                U8Arr net_val_init_u8, int32_t src_i, int32_t v0, int32_t v1,
                                I64Arr pw_ps_array, int64_t sim_end_ps, int64_t eps_ps,
                                bool delay_policy_is_max, bool vcs_align) {
  std::vector<py::object> keep;
  const ConeView cv = cone_view_of_packed(packed_obj, keep);
  const DelayView dv = delay_view_of(delaytab_obj, keep);
  const int32_t n_nets = (int32_t)cv.rec_idx.n;

  auto init_info = net_val_init_u8.request();
  if (init_info.ndim != 1) throw std::runtime_error("net_val_init_u8 must be 1-D");
  if ((int64_t)init_info.shape[0] != n_nets)
    throw std::runtime_error("net_val_init_u8 length does not match the cone's net count");
  const uint8_t* init_p = static_cast<const uint8_t*>(init_info.ptr);

  auto pw_info = pw_ps_array.request();
  if (pw_info.ndim != 1) throw std::runtime_error("pw_ps_array must be 1-D");
  const int64_t n_pw = (int64_t)pw_info.shape[0];
  const int64_t* pw_ps_p = static_cast<const int64_t*>(pw_info.ptr);

  py::list result;
  RecEvents rec_events;
  simulate_pulses_core(cv, dv, init_p, src_i, v0, v1, pw_ps_p, n_pw, sim_end_ps, eps_ps,
                       delay_policy_is_max, vcs_align, rec_events, [&](int64_t, const SimStats& stats) {
    // flatten the recorded events (CSR over record slots)
    const int32_t n_rec_slots = (int32_t)rec_events.size();
    I32Arr ev_ptr({(py::ssize_t)(n_rec_slots + 1)});
    int32_t* ptrp = ev_ptr.mutable_data();
    ptrp[0] = 0;
    int64_t total = 0;
    for (int32_t i = 0; i < n_rec_slots; i++) {
      total += (int64_t)rec_events[(size_t)i].size();
      ptrp[i + 1] = (int32_t)total;
    }
    I64Arr ev_t({(py::ssize_t)total});
    U8Arr ev_v({(py::ssize_t)total});
    int64_t* tp = ev_t.mutable_data();
    uint8_t* vp = ev_v.mutable_data();
    int64_t pos = 0;
    for (int32_t i = 0; i < n_rec_slots; i++) {
      for (auto& pr : rec_events[(size_t)i]) {
        tp[pos] = pr.first;
        vp[pos] = (uint8_t)(pr.second & 1);
        pos++;
      }
    }
    result.append(py::make_tuple(ev_ptr, ev_t, ev_v, stats_dict(stats)));
  });
  return result;
}

// Walk one slot's event stream and call emit(t_enter, t_exit) for every
// excursion away from `base`.  An excursion that returns at the same instant is
// not a pulse; one still open at the end gets t_exit = inf_ps.
template <typename Emit>
static inline void for_each_pulse(const int64_t* et, const uint8_t* ev, int32_t a, int32_t b,
                                  int base, int64_t inf_ps, Emit&& emit) {
  int cur = base;
  bool in_pulse = false;
  int64_t t_enter = 0;
  for (int32_t k = a; k < b; ++k) {
    const int64_t t_ps = et[k];
    const int v = (int)(ev[k] & 1);
    if (!in_pulse) {
      if (cur == base && v != base) {
        in_pulse = true;
        t_enter = t_ps;
      }
      cur = v;
    } else {
      if (v == base) {
        if (t_ps > t_enter) emit(t_enter, t_ps);
        in_pulse = false;
      }
      cur = v;
    }
  }
  if (in_pulse) emit(t_enter, inf_ps);
}

// emit(dpi, t_enter, t_exit) for every pulse at every recorded D net dpi, in
// D-net order (shared by extract_pulses and run_block)
template <typename Emit>
static inline void for_each_dp_pulse(const int32_t* eptr, int32_t n_rec_ptr, const int64_t* et,
                                     const uint8_t* ev, const int32_t* slot, const uint8_t* basep,
                                     int32_t n_dp, int64_t inf_ps, Emit&& emit) {
  for (int32_t dpi = 0; dpi < n_dp; ++dpi) {
    const int32_t rs = slot[dpi];
    if (rs < 0 || rs + 1 >= n_rec_ptr) continue;
    const int32_t a = eptr[rs], b = eptr[rs + 1];
    if (a >= b) continue;
    for_each_pulse(et, ev, a, b, basep[dpi] & 1, inf_ps,
                   [&](int64_t ta, int64_t tb) { emit(dpi, ta, tb); });
  }
}

// ---------------------------------------------------------------------------
// extract_pulses: pulse intervals at every recorded D net
// ---------------------------------------------------------------------------
static py::tuple extract_pulses(I32Arr ev_ptr, I64Arr ev_t, U8Arr ev_v, I32Arr dp_rec_slot,
                                U8Arr dp_base_u8, int64_t inf_ps) {
  if (ev_ptr.ndim() != 1 || ev_t.ndim() != 1 || ev_v.ndim() != 1)
    throw std::runtime_error("ev_ptr/ev_t/ev_v must be 1-D");
  if (dp_rec_slot.ndim() != 1 || dp_base_u8.ndim() != 1 || dp_rec_slot.shape(0) != dp_base_u8.shape(0))
    throw std::runtime_error("dp_rec_slot/dp_base_u8 must be 1-D and of equal length");
  const int32_t n_dp = (int32_t)dp_rec_slot.shape(0);
  const int32_t n_rec_ptr = (int32_t)ev_ptr.shape(0);
  const int32_t* eptr = ev_ptr.data();
  const int64_t* et = ev_t.data();
  const uint8_t* ev = ev_v.data();
  const int32_t* slot = dp_rec_slot.data();
  const uint8_t* basep = dp_base_u8.data();

  std::vector<int32_t> o_dp;
  std::vector<int64_t> o_a, o_b;
  for_each_dp_pulse(eptr, n_rec_ptr, et, ev, slot, basep, n_dp, inf_ps,
                    [&](int32_t dpi, int64_t ta, int64_t tb) {
    o_dp.push_back(dpi);
    o_a.push_back(ta);
    o_b.push_back(tb);
  });
  const py::ssize_t n = (py::ssize_t)o_dp.size();
  I32Arr r_dp({n});
  I64Arr r_a({n}), r_b({n});
  std::copy(o_dp.begin(), o_dp.end(), r_dp.mutable_data());
  std::copy(o_a.begin(), o_a.end(), r_a.mutable_data());
  std::copy(o_b.begin(), o_b.end(), r_b.mutable_data());
  return py::make_tuple(r_dp, r_a, r_b);
}

// ---------------------------------------------------------------------------
// run_block: every (cycle, site, pulse width) trial of a block of cycles
// ---------------------------------------------------------------------------
// The trial loop of setfi.campaign.run, in C++: for each site and cycle, bind
// the recorded state to the compiled cone, run simulate_pulses_core for all
// pulse widths and turn the D-pin events into records (as extract_pulses).
// Sites are shared out over n_threads threads, the calling thread among them;
// the result does not depend on the thread count.  A Python signal handler that
// raises (Ctrl-C) stops the block within about 50 ms plus one site's trials.

// The propagation options of a campaign: the keys of run_block's `options`
// (setfi.campaign.run.engine_options), i.e. the trailing arguments of
// simulate_pulses.  Unknown or missing keys are an error.
struct EngineOptions {
  int64_t sim_end_ps = 0;
  int64_t eps_ps = 0;
  bool delay_policy_is_max = true;
  bool vcs_align = false;
};

static EngineOptions engine_options_from(const py::dict& d) {
  static const char* const keys[] = {"sim_end_ps", "eps_ps", "delay_policy_is_max", "vcs_align"};
  for (auto kv : d) {
    const std::string k = py::cast<std::string>(kv.first);
    if (std::find_if(std::begin(keys), std::end(keys), [&](const char* x) { return k == x; }) == std::end(keys))
      throw std::runtime_error("run_block: unknown engine option '" + k + "'");
  }
  auto need = [&](const char* k) -> py::object {
    if (!d.contains(k)) throw std::runtime_error(std::string("run_block: engine option '") + k + "' is missing");
    return d[k];
  };
  EngineOptions o;
  o.sim_end_ps = py::cast<int64_t>(need("sim_end_ps"));
  o.eps_ps = py::cast<int64_t>(need("eps_ps"));
  o.delay_policy_is_max = py::cast<bool>(need("delay_policy_is_max"));
  o.vcs_align = py::cast<bool>(need("vcs_align"));
  return o;
}

template <typename T>
static ArrView<T> view_vec(const std::vector<T>& v) {
  ArrView<T> a;
  a.p = v.data();
  a.n = (int64_t)v.size();
  return a;
}

static ConeView cone_view_of(const setfi_fast::CompiledCone& c) {
  ConeView cv;
  cv.fanout_ptr = view_vec(c.fanout_ptr);
  cv.fanout_idx = view_vec(c.fanout_idx);
  cv.gate_out = view_vec(c.gate_out);
  cv.gate_var_ptr = view_vec(c.gate_var_ptr);
  cv.gate_var_nets = view_vec(c.gate_var_nets);
  cv.gate_truth_ptr = view_vec(c.gate_truth_ptr);
  cv.gate_truth = view_vec(c.gate_truth);
  cv.arc_dkey_id = view_vec(c.arc_dkey_id);
  cv.arc_cause_i = view_vec(c.arc_cause_i);
  cv.arc_side_mode = view_vec(c.arc_side_mode);
  cv.arc_side_ptr = view_vec(c.arc_side_ptr);
  cv.arc_side_nets = view_vec(c.arc_side_nets);
  cv.rec_idx = view_vec(c.rec_idx);
  cv.has_wires = !c.wire_sink.empty();
  if (cv.has_wires) {
    cv.wire_ptr = view_vec(c.wire_ptr);
    cv.wire_zsplit = view_vec(c.wire_zsplit);
    cv.wire_sink = view_vec(c.wire_sink);
    cv.wire_rise_ps = view_vec(c.wire_rise_ps);
    cv.wire_fall_ps = view_vec(c.wire_fall_ps);
  }
  return cv;
}

// per-site counters returned by run_block (columns of "site_stats"; the
// order of setfi.campaign.run.SITE_STATS)
enum SiteStat {
  ST_N_SIMS, ST_APPLIED, ST_C_REPLACE, ST_C_BACK, ST_KEPT_SAME, ST_COAL_NETS, ST_COAL_EVENTS,
  ST_DROP_NO_DELAY, ST_DROP_NO_CAND, ST_MAX_HEAP, ST_REACHING_D, ST_WITH_RECORDS, ST_N
};

// The cycle and width_ps columns of a record are int32 (setfi.records.COLUMNS).
// As in the Python trial loop with a current NumPy (np.full(..., np.int32)
// raises in NumPy 2.2 and later; NumPy 2.0 and earlier wrap the value), a value
// outside that range is an error only when a record would hold it.
static int32_t record_int32(int64_t v, const char* what) {
  if (v < std::numeric_limits<int32_t>::min() || v > std::numeric_limits<int32_t>::max())
    throw std::overflow_error("Python integer " + std::to_string(v) + " out of bounds for int32 (" + what + ")");
  return (int32_t)v;
}

// what one thread produced, plus its scratch space
struct BlockThreadOut {
  std::vector<int32_t> cycle, site, width, ff;
  std::vector<uint8_t> base;
  std::vector<int64_t> t_enter, t_exit;
  std::vector<uint8_t> net_val, dp_base;
  std::vector<int32_t> eptr;
  std::vector<int64_t> et;
  std::vector<uint8_t> ev;
  RecEvents rec_events;
  std::exception_ptr error;
  int64_t error_item = -1;
};

static py::dict run_block(py::list cones_list,
                          py::array_t<uint8_t, py::array::c_style | py::array::forcecast> states,
                          I64Arr cycles_arr, I32Arr site_ids_arr, I64Arr pw_ps_array,
                          py::object delaytab_obj, py::dict options, int64_t inf_ps, int n_threads) {
  const EngineOptions opt = engine_options_from(options);
  std::vector<py::object> keep;
  const DelayView dv = delay_view_of(delaytab_obj, keep);

  const int64_t n_sites = (int64_t)py::len(cones_list);
  std::vector<const setfi_fast::CompiledCone*> cones((size_t)n_sites);
  for (int64_t j = 0; j < n_sites; j++)
    cones[(size_t)j] = &py::cast<const setfi_fast::CompiledCone&>(cones_list[(size_t)j]);

  if (states.ndim() != 2) throw std::runtime_error("run_block: states must be [n_cycles, n_nets]");
  if (cycles_arr.ndim() != 1 || site_ids_arr.ndim() != 1 || pw_ps_array.ndim() != 1)
    throw std::runtime_error("run_block: cycles, site_ids and pulse widths must be 1-D");
  const int64_t n_cyc = (int64_t)cycles_arr.shape(0);
  if ((int64_t)states.shape(0) != n_cyc) throw std::runtime_error("run_block: one state row per cycle");
  if ((int64_t)site_ids_arr.shape(0) != n_sites) throw std::runtime_error("run_block: one site id per cone");
  const int64_t n_glob = (int64_t)states.shape(1);
  const uint8_t* states_p = states.data();
  const int64_t* cycles = cycles_arr.data();
  const int32_t* site_ids = site_ids_arr.data();
  const int64_t n_pw = (int64_t)pw_ps_array.shape(0);
  const int64_t* pws = pw_ps_array.data();

  int nt = n_threads < 1 ? 1 : n_threads;
  if ((int64_t)nt > n_sites) nt = (int)std::max<int64_t>(1, n_sites);

  // per item (cycle k, site j), k-major: where its records are
  const size_t n_items = (size_t)(n_cyc * n_sites);
  std::vector<int64_t> rec_start(n_items, 0), rec_count(n_items, 0);
  std::vector<int32_t> thread_of((size_t)n_sites, 0);
  std::vector<int64_t> site_stats((size_t)n_sites * ST_N, 0);
  std::vector<BlockThreadOut> outs((size_t)nt);
  // largest cones first, so that no thread is left with a big one at the end;
  // the processing order does not affect the result
  std::vector<int64_t> order((size_t)n_sites);
  for (int64_t j = 0; j < n_sites; j++) order[(size_t)j] = j;
  if (nt > 1) {
    auto size_of = [&](int64_t j) {
      return (int64_t)cones[(size_t)j]->gate_var_nets.size() + (int64_t)cones[(size_t)j]->rec_idx.size();
    };
    std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) { return size_of(a) > size_of(b); });
  }
  std::atomic<int64_t> next_site{0};
  std::atomic<bool> stop{false};
  bool interrupted = false;  // a Python signal handler raised (set by thread 0, the caller)
  setfi_fast::SignalPoll poll;

  auto work = [&](int tid) {
    BlockThreadOut& o = outs[(size_t)tid];
    for (;;) {
      if (stop.load()) return;
      if (tid == 0 && poll.pending()) {
        interrupted = true;
        stop.store(true);
        return;
      }
      const int64_t q = next_site.fetch_add(1);
      if (q >= n_sites) return;
      const int64_t j = order[(size_t)q];
      thread_of[(size_t)j] = tid;
      const setfi_fast::CompiledCone& c = *cones[(size_t)j];
      const ConeView cv = cone_view_of(c);
      const int32_t n_ff = c.n_ff();
      const int32_t n_dp = (int32_t)c.dp_rec_slot.size();
      int64_t* S = &site_stats[(size_t)j * ST_N];
      for (int64_t k = 0; k < n_cyc; k++) {
        const size_t item = (size_t)(k * n_sites + j);
        rec_start[item] = (int64_t)o.cycle.size();
        try {
          setfi_fast::bind_state(c, states_p + (size_t)(k * n_glob), n_glob, o.net_val, o.dp_base);
          const int32_t v0 = (int32_t)(o.net_val[(size_t)c.src_i] & 1);
          simulate_pulses_core(
              cv, dv, o.net_val.data(), c.src_i, v0, v0 ^ 1, pws, n_pw, opt.sim_end_ps, opt.eps_ps,
              opt.delay_policy_is_max, opt.vcs_align, o.rec_events, [&](int64_t pw_idx, const SimStats& st) {
            S[ST_N_SIMS] += 1;
            S[ST_APPLIED] += st.applied_events;
            S[ST_C_REPLACE] += st.canceled_replace;
            S[ST_C_BACK] += st.canceled_back_to_current;
            S[ST_KEPT_SAME] += st.kept_same_target;
            S[ST_COAL_NETS] += st.coalesced_nets;
            S[ST_COAL_EVENTS] += st.coalesced_events;
            S[ST_DROP_NO_DELAY] += st.dropped_arc_no_delay;
            S[ST_DROP_NO_CAND] += st.dropped_no_candidate_arc;
            S[ST_MAX_HEAP] = std::max(S[ST_MAX_HEAP], st.max_heap);
            if (n_ff == 0 || n_dp == 0) return;
            // the CSR event stream simulate_pulses returns
            const int32_t n_slots = (int32_t)o.rec_events.size();
            o.eptr.resize((size_t)n_slots + 1);
            o.eptr[0] = 0;
            int64_t total = 0;
            for (int32_t i = 0; i < n_slots; i++) {
              total += (int64_t)o.rec_events[(size_t)i].size();
              o.eptr[(size_t)i + 1] = (int32_t)total;
            }
            if (total == 0) return;
            S[ST_REACHING_D] += 1;
            o.et.resize((size_t)total);
            o.ev.resize((size_t)total);
            int64_t pos = 0;
            for (int32_t i = 0; i < n_slots; i++) {
              for (auto& pr : o.rec_events[(size_t)i]) {
                o.et[(size_t)pos] = pr.first;
                o.ev[(size_t)pos] = (uint8_t)(pr.second & 1);
                pos++;
              }
            }
            // one record per (pulse at a D net, observed FF on that net)
            int32_t cyc32 = 0, pw32 = 0;
            int64_t n_rec = 0;
            for_each_dp_pulse(o.eptr.data(), n_slots + 1, o.et.data(), o.ev.data(), c.dp_rec_slot.data(),
                              o.dp_base.data(), n_dp, inf_ps, [&](int32_t dpi, int64_t ta, int64_t tb) {
              for (int32_t kk = c.dp_ptr[(size_t)dpi]; kk < c.dp_ptr[(size_t)dpi + 1]; kk++) {
                if (n_rec == 0) {
                  cyc32 = record_int32(cycles[k], "cycle");
                  pw32 = record_int32(pws[pw_idx], "pulse width");
                }
                o.cycle.push_back(cyc32);
                o.site.push_back(site_ids[j]);
                o.width.push_back(pw32);
                o.ff.push_back(c.ffids[(size_t)c.dp_idx[(size_t)kk]]);
                o.base.push_back(o.dp_base[(size_t)dpi]);
                o.t_enter.push_back(ta);
                o.t_exit.push_back(tb);
                n_rec++;
              }
            });
            if (n_rec > 0) S[ST_WITH_RECORDS] += 1;
          });
        } catch (...) {
          if (!o.error || (int64_t)item < o.error_item) {
            o.error = std::current_exception();
            o.error_item = (int64_t)item;
          }
          rec_count[item] = 0;
          break;
        }
        rec_count[item] = (int64_t)o.cycle.size() - rec_start[item];
      }
    }
  };

  {
    py::gil_scoped_release release;
    setfi_fast::run_workers(nt, stop, work);
  }
  if (interrupted) throw py::error_already_set();
  {
    // the error the sequential loop would have met first
    int64_t first = -1;
    std::exception_ptr err;
    for (auto& o : outs)
      if (o.error && (first < 0 || o.error_item < first)) {
        first = o.error_item;
        err = o.error;
      }
    if (err) std::rethrow_exception(err);
  }

  // gather in k-major order, the order of the Python trial loop
  int64_t n_total = 0;
  for (size_t it = 0; it < n_items; it++) n_total += rec_count[it];
  py::array_t<int32_t> a_cycle((py::ssize_t)n_total), a_site((py::ssize_t)n_total),
      a_width((py::ssize_t)n_total), a_ff((py::ssize_t)n_total);
  py::array_t<uint8_t> a_base((py::ssize_t)n_total);
  py::array_t<int64_t> a_te((py::ssize_t)n_total), a_tx((py::ssize_t)n_total);
  py::array_t<int64_t> a_off((py::ssize_t)(n_items + 1));
  {
    int32_t* pc = a_cycle.mutable_data();
    int32_t* ps = a_site.mutable_data();
    int32_t* pw = a_width.mutable_data();
    int32_t* pf = a_ff.mutable_data();
    uint8_t* pb = a_base.mutable_data();
    int64_t* pe = a_te.mutable_data();
    int64_t* px = a_tx.mutable_data();
    int64_t* po = a_off.mutable_data();
    py::gil_scoped_release release;
    int64_t pos = 0;
    po[0] = 0;
    for (int64_t k = 0; k < n_cyc; k++) {
      for (int64_t j = 0; j < n_sites; j++) {
        const size_t it = (size_t)(k * n_sites + j);
        const BlockThreadOut& o = outs[(size_t)thread_of[(size_t)j]];
        const size_t a = (size_t)rec_start[it], n = (size_t)rec_count[it];
        std::copy(o.cycle.begin() + a, o.cycle.begin() + a + n, pc + pos);
        std::copy(o.site.begin() + a, o.site.begin() + a + n, ps + pos);
        std::copy(o.width.begin() + a, o.width.begin() + a + n, pw + pos);
        std::copy(o.ff.begin() + a, o.ff.begin() + a + n, pf + pos);
        std::copy(o.base.begin() + a, o.base.begin() + a + n, pb + pos);
        std::copy(o.t_enter.begin() + a, o.t_enter.begin() + a + n, pe + pos);
        std::copy(o.t_exit.begin() + a, o.t_exit.begin() + a + n, px + pos);
        pos += (int64_t)n;
        po[it + 1] = pos;
      }
    }
  }
  outs.clear();
  outs.shrink_to_fit();

  py::array_t<int64_t> a_stats({(py::ssize_t)n_sites, (py::ssize_t)ST_N});
  std::copy(site_stats.begin(), site_stats.end(), a_stats.mutable_data());

  py::dict r;
  r["cycle"] = a_cycle;
  r["site"] = a_site;
  r["width_ps"] = a_width;
  r["ff"] = a_ff;
  r["base"] = a_base;
  r["t_enter"] = a_te;
  r["t_exit"] = a_tx;
  r["rec_off"] = a_off;
  r["site_stats"] = a_stats;
  return r;
}

}  // namespace

PYBIND11_MODULE(_engine, m) {
  m.doc() = "setfi event-driven SET propagation and capture kernel";

  m.def("simulate_pulses", &simulate_pulses, py::arg("packed"), py::arg("delaytab"),
        py::arg("net_val_init_u8"), py::arg("src_i"), py::arg("v0"), py::arg("v1"),
        py::arg("pw_ps_array"), py::arg("sim_end_ps"), py::arg("eps_ps"),
        py::arg("delay_policy_is_max"), py::arg("vcs_align") = false,
        R"doc(Inject one SET per pulse width on net src_i (flip v0 -> v1 at t=0,
back to v0 at t=pw) and propagate it through the packed cone until sim_end_ps.

eps_ps: events up to eps_ps after the earliest pending one are applied
together with it (coalesce_window_ps; 0: only simultaneous events).
delay_policy_is_max: when several inputs of a gate change at once and none has
an active delay path, use the largest (True) or smallest default delay of the
inputs that explain the output change (propagation.delay_choice; see
docs/MODEL.md).  vcs_align: propagation.vcs_align.

Returns a list with one (ev_ptr, ev_t, ev_v, stats) tuple per pulse width:
ev_ptr is a CSR pointer over record slots, ev_t / ev_v the time (ps) and new
value of every transition applied to a recorded net.)doc");

  m.def("extract_pulses", &extract_pulses, py::arg("ev_ptr"), py::arg("ev_t"), py::arg("ev_v"),
        py::arg("dp_rec_slot"), py::arg("dp_base_u8"), py::arg("inf_ps"),
        R"doc(Pulse intervals at each recorded D net.

For each D net dpi (record slot dp_rec_slot[dpi], fault-free value
dp_base_u8[dpi]), every excursion away from the fault-free value becomes one
interval [t_enter, t_exit) in ps after the SET start; a pulse still open when the
simulation stops gets t_exit = inf_ps.  Returns (dp_index, t_enter, t_exit).)doc");

  m.def("run_block", &run_block, py::arg("cones"), py::arg("states"), py::arg("cycles"),
        py::arg("site_ids"), py::arg("pw_ps_array"), py::arg("delaytab"), py::arg("options"),
        py::arg("inf_ps"), py::arg("n_threads") = 1,
        R"doc(All trials of a block of cycles: for every cone (a CompiledCone), every
cycle (row of states, one byte per global net) and every pulse width, bind the
state, propagate, and collect the pulse records.  options holds the propagation
flags, the trailing arguments of simulate_pulses by name: sim_end_ps, eps_ps,
delay_policy_is_max and vcs_align.

Returns a dict: the record columns (cycle, site, width_ps, ff, base, t_enter,
t_exit) in the order of a cycle-major loop over (cycle, cone, width);
rec_off[k * n_cones + j] .. rec_off[k * n_cones + j + 1] are the records of
cycle k and cone j; site_stats[j] the trial counters of cone j.)doc");

  setfi_fast::register_cone_compiler(m);
  setfi_fast::register_patterns(m);
  setfi_fast::register_logic(m);
}
