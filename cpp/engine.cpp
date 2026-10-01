// setfi._engine — event-driven SET pulse propagation and D-pin capture.
//
// Three entry points:
//   simulate_pulses   inject one SET per pulse width on one source net and
//                     propagate it through a packed fan-out cone; returns the
//                     event stream seen at every recorded FF D net.
//   extract_pulses    turn a recorded event stream into pulse intervals
//                     [t_enter, t_exit) relative to the SET start.
//   capture_masks     for a grid of SET start times, which FFs capture a wrong
//                     value (pulse overlaps the FF's setup/hold window).
//
// The propagation model (see docs/MODEL.md):
//   * logic: every gate is re-evaluated from its truth table on each input
//     change (logical masking);
//   * delay: the output transition takes the SDF IOPATH delay of the input
//     that explains the change, conditional on side-input values where the SDF
//     gives COND entries; when several inputs could explain it, the max (or
//     min) delay is used;
//   * electrical masking (inertial delay): a pending output transition is
//     cancelled when its causing input reverts before t_cause + delay + margin;
//   * one pending event per gate output; a new evaluation that returns the
//     output to its current value cancels the pending one;
//   * optional transport wires (SDF INTERCONNECT) delay but never filter.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

struct SimStats {
  int64_t applied_events = 0;
  int64_t canceled_late = 0;
  int64_t canceled_race = 0;
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
using U64Arr = py::array_t<uint64_t, py::array::c_style | py::array::forcecast>;

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
static inline int64_t delay_get_ps(const ArrView<int64_t>& default_rise_ps,
                                   const ArrView<int64_t>& default_fall_ps,
                                   const ArrView<int32_t>& patt_ptr,
                                   const ArrView<int64_t>& patt_mask,
                                   const ArrView<int64_t>& patt_valbits,
                                   const ArrView<int64_t>& patt_dt_ps,
                                   const ArrView<uint8_t>& patt_is_rise,
                                   int32_t dkey_id, int tr_is_rise, bool sbits_valid,
                                   int64_t sbits) {
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
  return def_ps;
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
  d["canceled_late"] = s.canceled_late;
  d["canceled_race"] = s.canceled_race;
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
static py::list simulate_pulses(py::object packed_obj, py::object delaytab_obj,
                                U8Arr net_val_init_u8, int32_t src_i, int32_t v0, int32_t v1,
                                I64Arr pw_ps_array, int64_t sim_end_ps, int64_t eps_ps,
                                bool enable_em, int64_t margin_ps, bool delay_policy_is_max) {
  auto fanout_ptr_a = as_1d_array<int32_t>(packed_obj.attr("fanout_ptr"), "fanout_ptr");
  auto fanout_idx_a = as_1d_array<int32_t>(packed_obj.attr("fanout_idx"), "fanout_idx");
  auto gate_out_a = as_1d_array<int32_t>(packed_obj.attr("gate_out"), "gate_out");
  auto gate_var_ptr_a = as_1d_array<int32_t>(packed_obj.attr("gate_var_ptr"), "gate_var_ptr");
  auto gate_var_nets_a = as_1d_array<int32_t>(packed_obj.attr("gate_var_nets"), "gate_var_nets");
  auto gate_truth_ptr_a = as_1d_array<int32_t>(packed_obj.attr("gate_truth_ptr"), "gate_truth_ptr");
  auto gate_truth_a = as_1d_array<uint8_t>(packed_obj.attr("gate_truth"), "gate_truth");
  auto arc_dkey_id_a = as_1d_array<int32_t>(packed_obj.attr("arc_dkey_id"), "arc_dkey_id");
  auto arc_cause_i_a = as_1d_array<int32_t>(packed_obj.attr("arc_cause_i"), "arc_cause_i");
  auto arc_side_mode_a = as_1d_array<int8_t>(packed_obj.attr("arc_side_mode"), "arc_side_mode");
  auto arc_side_ptr_a = as_1d_array<int32_t>(packed_obj.attr("arc_side_ptr"), "arc_side_ptr");
  auto arc_side_nets_a = as_1d_array<int32_t>(packed_obj.attr("arc_side_nets"), "arc_side_nets");
  auto rec_idx_a = as_1d_array<int32_t>(packed_obj.attr("rec_idx"), "rec_idx");

  auto default_rise_a = as_1d_array<int64_t>(delaytab_obj.attr("default_rise_ps"), "default_rise_ps");
  auto default_fall_a = as_1d_array<int64_t>(delaytab_obj.attr("default_fall_ps"), "default_fall_ps");
  auto patt_ptr_a = as_1d_array<int32_t>(delaytab_obj.attr("patt_ptr"), "patt_ptr");
  auto patt_mask_a = as_1d_array<int64_t>(delaytab_obj.attr("patt_mask"), "patt_mask");
  auto patt_valbits_a = as_1d_array<int64_t>(delaytab_obj.attr("patt_valbits"), "patt_valbits");
  auto patt_dt_ps_a = as_1d_array<int64_t>(delaytab_obj.attr("patt_dt_ps"), "patt_dt_ps");
  auto patt_is_rise_a = as_1d_array<uint8_t>(delaytab_obj.attr("patt_is_rise"), "patt_is_rise");

  auto fanout_ptr = view_1d(fanout_ptr_a, "fanout_ptr");
  auto fanout_idx = view_1d(fanout_idx_a, "fanout_idx");
  auto gate_out = view_1d(gate_out_a, "gate_out");
  auto gate_var_ptr = view_1d(gate_var_ptr_a, "gate_var_ptr");
  auto gate_var_nets = view_1d(gate_var_nets_a, "gate_var_nets");
  auto gate_truth_ptr = view_1d(gate_truth_ptr_a, "gate_truth_ptr");
  auto gate_truth = view_1d(gate_truth_a, "gate_truth");
  auto arc_dkey_id = view_1d(arc_dkey_id_a, "arc_dkey_id");
  auto arc_cause_i = view_1d(arc_cause_i_a, "arc_cause_i");
  auto arc_side_mode = view_1d(arc_side_mode_a, "arc_side_mode");
  auto arc_side_ptr = view_1d(arc_side_ptr_a, "arc_side_ptr");
  auto arc_side_nets = view_1d(arc_side_nets_a, "arc_side_nets");
  auto rec_idx = view_1d(rec_idx_a, "rec_idx");

  // Optional transport wires (SDF INTERCONNECT), CSR over driver nets, sorted
  // zero-delay first: [wire_ptr[n], wire_zsplit[n]) have d == 0 and are
  // propagated inside the current time batch; [wire_zsplit[n], wire_ptr[n+1])
  // have d > 0 and become one heap event per edge.
  ArrView<int32_t> wire_ptr, wire_zsplit, wire_sink;
  ArrView<int64_t> wire_rise_ps, wire_fall_ps;
  I32Arr wp_keep, wz_keep, ws_keep;
  I64Arr wr_keep, wf_keep;
  bool has_wires = false;
  if (py::hasattr(packed_obj, "wire_sink")) {
    wp_keep = as_1d_array<int32_t>(packed_obj.attr("wire_ptr"), "wire_ptr");
    wz_keep = as_1d_array<int32_t>(packed_obj.attr("wire_zsplit"), "wire_zsplit");
    ws_keep = as_1d_array<int32_t>(packed_obj.attr("wire_sink"), "wire_sink");
    wr_keep = as_1d_array<int64_t>(packed_obj.attr("wire_rise_ps"), "wire_rise_ps");
    wf_keep = as_1d_array<int64_t>(packed_obj.attr("wire_fall_ps"), "wire_fall_ps");
    wire_ptr = view_1d(wp_keep, "wire_ptr");
    wire_zsplit = view_1d(wz_keep, "wire_zsplit");
    wire_sink = view_1d(ws_keep, "wire_sink");
    wire_rise_ps = view_1d(wr_keep, "wire_rise_ps");
    wire_fall_ps = view_1d(wf_keep, "wire_fall_ps");
    if (wire_ptr.n != rec_idx.n + 1 || wire_zsplit.n != rec_idx.n)
      throw std::runtime_error("wire_ptr/wire_zsplit length mismatch with n_nets");
    if (wire_rise_ps.n != wire_sink.n || wire_fall_ps.n != wire_sink.n)
      throw std::runtime_error("wire_sink/wire_rise_ps/wire_fall_ps length mismatch");
    has_wires = (wire_sink.n > 0);
  }

  auto default_rise = view_1d(default_rise_a, "default_rise_ps");
  auto default_fall = view_1d(default_fall_a, "default_fall_ps");
  auto patt_ptr = view_1d(patt_ptr_a, "patt_ptr");
  auto patt_mask = view_1d(patt_mask_a, "patt_mask");
  auto patt_valbits = view_1d(patt_valbits_a, "patt_valbits");
  auto patt_dt_ps = view_1d(patt_dt_ps_a, "patt_dt_ps");
  auto patt_is_rise = view_1d(patt_is_rise_a, "patt_is_rise");

  const int32_t n_nets = (int32_t)rec_idx.n;
  const int32_t n_gates = (int32_t)gate_out.n;

  auto init_info = net_val_init_u8.request();
  if (init_info.ndim != 1) throw std::runtime_error("net_val_init_u8 must be 1-D");
  if ((int64_t)init_info.shape[0] != n_nets)
    throw std::runtime_error("net_val_init_u8 length does not match the cone's net count");
  const uint8_t* init_p = static_cast<const uint8_t*>(init_info.ptr);

  int32_t n_rec_slots = 0;
  for (int32_t i = 0; i < n_nets; i++) {
    const int32_t r = rec_idx.p[i];
    if (r >= 0) n_rec_slots = std::max(n_rec_slots, (int32_t)(r + 1));
  }

  auto pw_info = pw_ps_array.request();
  if (pw_info.ndim != 1) throw std::runtime_error("pw_ps_array must be 1-D");
  const int64_t n_pw = (int64_t)pw_info.shape[0];
  const int64_t* pw_ps_p = static_cast<const int64_t*>(pw_info.ptr);

  py::list result;
  std::vector<std::vector<std::pair<int64_t, uint8_t>>> rec_events((size_t)n_rec_slots);

  for (int64_t pw_idx = 0; pw_idx < n_pw; ++pw_idx) {
    const int64_t pw_ps = pw_ps_p[pw_idx];
    SimStats stats;

    if (src_i < 0) {
      I32Arr ev_ptr({(py::ssize_t)(n_rec_slots + 1)});
      std::fill(ev_ptr.mutable_data(), ev_ptr.mutable_data() + (n_rec_slots + 1), 0);
      I64Arr ev_t({(py::ssize_t)0});
      U8Arr ev_v({(py::ssize_t)0});
      result.append(py::make_tuple(ev_ptr, ev_t, ev_v, stats_dict(stats)));
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
    std::vector<int32_t> tok2cause(2, -1);
    std::vector<int64_t> tok2deadline(2, std::numeric_limits<int64_t>::min());

    auto ensure_tok = [&](int32_t tok) {
      if (tok >= (int32_t)canceled.size()) {
        const size_t new_sz = (size_t)tok + 1;
        canceled.resize(new_sz, 0);
        tok2outnet.resize(new_sz, -1);
        tok2pend_val.resize(new_sz, 0);
        tok2cause.resize(new_sz, -1);
        tok2deadline.resize(new_sz, std::numeric_limits<int64_t>::min());
      }
    };

    std::vector<std::vector<std::pair<int64_t, int32_t>>> pending_by_cause;
    if (enable_em) pending_by_cause.resize((size_t)n_nets);
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

    enum CancelWhy { WHY_LATE, WHY_RACE, WHY_REPLACE, WHY_BACK };
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
      tok2cause[(size_t)tok] = -1;
      tok2deadline[(size_t)tok] = std::numeric_limits<int64_t>::min();
      if (why == WHY_LATE) stats.canceled_late++;
      else if (why == WHY_RACE) stats.canceled_race++;
      else if (why == WHY_REPLACE) stats.canceled_replace++;
    };

    // electrical masking: the cause net reverted before the deadline
    auto cancel_pending_due_to_input_revert = [&](int32_t cause_i, int64_t t_now_ps) {
      if (!enable_em) return;
      if (cause_i < 0 || cause_i >= n_nets) return;
      auto& lst = pending_by_cause[(size_t)cause_i];
      if (lst.empty()) return;
      for (auto& pr : lst) {
        if (t_now_ps < pr.first) cancel_token(pr.second, WHY_LATE);
      }
      lst.clear();
    };

    // the SET: flip the source net at t=0, restore it at t=pw
    push_event(0, src_i, v1);
    push_event(pw_ps, src_i, v0);

    std::vector<int32_t> best_tok((size_t)n_nets, -1);
    std::vector<uint8_t> best_val((size_t)n_nets, 0);
    std::vector<int32_t> touched;
    touched.reserve(1024);

    // A wire delays a pulse and never masks one: these helpers never touch the
    // pending/cause bookkeeping, so a wire cannot cancel anything.
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

    std::vector<int32_t> will_mark((size_t)n_nets, 0);
    int32_t will_tag = 1;

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

      will_tag++;
      if (will_tag >= 1'000'000'000) {
        std::fill(will_mark.begin(), will_mark.end(), 0);
        will_tag = 1;
      }
      bool any_will = false;
      for (int32_t ni : touched) {
        const uint8_t vv = best_val[(size_t)ni] & 1;
        if (vv != (net_val[(size_t)ni] & 1)) {
          will_mark[(size_t)ni] = will_tag;
          any_will = true;
        }
      }

      // same-timestamp race: an output event whose cause changes in this very
      // batch, before its deadline, is electrically masked
      if (enable_em && any_will) {
        bool any_canceled = false;
        for (int32_t ni : touched) {
          const int32_t tok = best_tok[(size_t)ni];
          if (tok <= 0) continue;
          ensure_tok(tok);
          const int32_t cause_i = tok2cause[(size_t)tok];
          if (cause_i < 0 || cause_i >= n_nets) continue;
          if (will_mark[(size_t)cause_i] != will_tag) continue;
          if (t0_ps < tok2deadline[(size_t)tok]) {
            cancel_token(tok, WHY_RACE);
            any_canceled = true;
          }
        }
        if (any_canceled) {
          for (int32_t ni : touched) best_tok[(size_t)ni] = -1;
          touched.clear();
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
        }
      }

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

      if (enable_em) {
        for (int32_t cn : changed_nets) cancel_pending_due_to_input_revert(cn, t0_ps);
      }
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

        if (y_new == y_cur) {
          const int32_t old_tok = pending_tok_by_outnet[(size_t)out_i];
          if (old_tok > 0) {
            ensure_tok(old_tok);
            if (!canceled[(size_t)old_tok] && tok2pend_val[(size_t)old_tok] != y_cur) {
              stats.canceled_back_to_current++;
              cancel_token(old_tok, WHY_BACK);
            }
          }
          continue;
        }

        const int tr_is_rise = (y_cur == 0 && y_new == 1) ? 1 : 0;

        // candidate inputs: changed inputs that alone explain the output change
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
        if (cands.empty()) continue;

        bool found = false;
        int64_t best_delay = 0;
        int32_t best_cause = -1;

        auto try_candidate = [&](int32_t ppos, int64_t& d_used, int32_t& cause_out, bool& ok) {
          const int32_t idx = va + ppos;
          const int32_t dkey_id = arc_dkey_id.p[idx];
          const int32_t cause_i = arc_cause_i.p[idx];
          // No delay arc for this input: it cannot carry the transition.
          if (dkey_id < 0 || cause_i < 0) {
            stats.dropped_arc_no_delay++;
            ok = false;
            return;
          }
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
                                patt_dt_ps, patt_is_rise, dkey_id, tr_is_rise, sbits_valid, sbits);
          cause_out = cause_i;
          ok = true;
        };

        if (delay_policy_is_max) {
          int64_t best_d = -1;
          for (int32_t pp : cands) {
            int64_t d_used = 0;
            int32_t cause_i = -1;
            bool ok = false;
            try_candidate(pp, d_used, cause_i, ok);
            if (!ok) continue;
            if (d_used > best_d) {
              best_d = d_used;
              best_delay = d_used;
              best_cause = cause_i;
              found = true;
            }
          }
        } else {
          int64_t best_d = (std::numeric_limits<int64_t>::max)();
          for (int32_t pp : cands) {
            int64_t d_used = 0;
            int32_t cause_i = -1;
            bool ok = false;
            try_candidate(pp, d_used, cause_i, ok);
            if (!ok) continue;
            if (d_used < best_d) {
              best_d = d_used;
              best_delay = d_used;
              best_cause = cause_i;
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

        if (enable_em) {
          const int64_t deadline = t0_ps + best_delay + margin_ps;
          if (best_cause >= 0 && best_cause < n_nets) {
            pending_by_cause[(size_t)best_cause].push_back({deadline, tok_new});
            tok2cause[(size_t)tok_new] = best_cause;
            tok2deadline[(size_t)tok_new] = deadline;
          }
        }
      }
    }

    // flatten the recorded events (CSR over record slots)
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
  }
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
  for (int32_t dpi = 0; dpi < n_dp; ++dpi) {
    const int32_t rs = slot[dpi];
    if (rs < 0 || rs + 1 >= n_rec_ptr) continue;
    const int32_t a = eptr[rs], b = eptr[rs + 1];
    if (a >= b) continue;
    for_each_pulse(et, ev, a, b, basep[dpi] & 1, inf_ps, [&](int64_t ta, int64_t tb) {
      o_dp.push_back(dpi);
      o_a.push_back(ta);
      o_b.push_back(tb);
    });
  }
  const py::ssize_t n = (py::ssize_t)o_dp.size();
  I32Arr r_dp({n});
  I64Arr r_a({n}), r_b({n});
  std::copy(o_dp.begin(), o_dp.end(), r_dp.mutable_data());
  std::copy(o_a.begin(), o_a.end(), r_a.mutable_data());
  std::copy(o_b.begin(), o_b.end(), r_b.mutable_data());
  return py::make_tuple(r_dp, r_a, r_b);
}

static inline int32_t lower_bound_i64(const int64_t* a, int32_t n, int64_t v) {
  int32_t l = 0, r = n;
  while (l < r) {
    const int32_t m = l + ((r - l) >> 1);
    if (a[m] < v) l = m + 1;
    else r = m;
  }
  return l;
}

static inline int32_t upper_bound_i64(const int64_t* a, int32_t n, int64_t v) {
  int32_t l = 0, r = n;
  while (l < r) {
    const int32_t m = l + ((r - l) >> 1);
    if (a[m] <= v) l = m + 1;
    else r = m;
  }
  return l;
}

static inline void require(bool cond, const char* msg) {
  if (!cond) throw std::runtime_error(msg);
}

// ---------------------------------------------------------------------------
// capture_masks: per SET start time, the bit set of FFs that capture
// ---------------------------------------------------------------------------
// A pulse [a, b) at the D pin (relative to the SET start s) is captured by an
// FF with sampling thresholds (thr_enter, thr_leave) iff
//     thr_leave - b + 1 <= s <= thr_enter - a - 1.
static py::array_t<uint64_t> capture_masks(I64Arr starts_ps, I64Arr thr_enter_local,
                                           I64Arr thr_leave_local, I32Arr ev_ptr, I64Arr ev_t,
                                           U8Arr ev_v, I32Arr dp_rec_slot, U8Arr dp_base_u8,
                                           I32Arr dp_localpos_ptr, I32Arr dp_localpos_idx,
                                           I32Arr word_index, U64Arr word_mask, int n_words,
                                           int64_t inf_ps) {
  require(starts_ps.ndim() == 1, "starts_ps must be 1-D");
  require(thr_enter_local.ndim() == 2 && thr_leave_local.ndim() == 2, "thr_*_local must be [2, n_ff]");
  require(ev_ptr.ndim() == 1 && ev_t.ndim() == 1 && ev_v.ndim() == 1, "ev_ptr/ev_t/ev_v must be 1-D");
  require(dp_rec_slot.ndim() == 1 && dp_base_u8.ndim() == 1, "dp_rec_slot/dp_base_u8 must be 1-D");
  require(dp_localpos_ptr.ndim() == 1 && dp_localpos_idx.ndim() == 1, "dp_localpos_ptr/idx must be 1-D");
  require(word_index.ndim() == 1 && word_mask.ndim() == 1, "word_index/word_mask must be 1-D");

  const int32_t n_st = (int32_t)starts_ps.shape(0);
  const int32_t n_rff = (int32_t)word_index.shape(0);
  require(word_mask.shape(0) == n_rff, "word_index and word_mask size mismatch");
  require(thr_enter_local.shape(0) == 2 && thr_leave_local.shape(0) == 2, "thr_*_local first dim must be 2");
  require(thr_enter_local.shape(1) == n_rff && thr_leave_local.shape(1) == n_rff,
          "thr_*_local second dim must match n_ff");
  const int32_t n_dp = (int32_t)dp_rec_slot.shape(0);
  require((int32_t)dp_base_u8.shape(0) == n_dp, "dp_base_u8 length must match n_dp");
  require((int32_t)dp_localpos_ptr.shape(0) == n_dp + 1, "dp_localpos_ptr must have length n_dp+1");
  require(n_words > 0, "n_words must be > 0");
  require(ev_t.shape(0) == ev_v.shape(0), "ev_t and ev_v must have the same length");

  py::array_t<uint64_t> out({(py::ssize_t)n_st, (py::ssize_t)n_words});
  uint64_t* outp = (uint64_t*)out.mutable_data();
  std::fill(outp, outp + (size_t)n_st * (size_t)n_words, 0ULL);
  if (n_st == 0 || n_rff == 0 || n_dp == 0) return out;

  const int64_t* starts = starts_ps.data();
  const int64_t starts0 = starts[0];
  const int64_t startsN = starts[n_st - 1];
  const int64_t* thr_enter = thr_enter_local.data();
  const int64_t* thr_leave = thr_leave_local.data();
  const int32_t* eptr = ev_ptr.data();
  const int64_t* et = ev_t.data();
  const uint8_t* ev = ev_v.data();
  const int32_t* dp_slot = dp_rec_slot.data();
  const uint8_t* dp_base = dp_base_u8.data();
  const int32_t* lp_ptr = dp_localpos_ptr.data();
  const int32_t* lp_idx = dp_localpos_idx.data();
  const int32_t* widx = word_index.data();
  const uint64_t* wmsk = word_mask.data();

  const size_t row_len = (size_t)n_st + 1u;
  static thread_local std::vector<int64_t> diffs;
  diffs.assign((size_t)n_rff * row_len, 0);

  {
    py::gil_scoped_release release;
    const int32_t n_rec_ptr = (int32_t)ev_ptr.shape(0);

    auto process_interval = [&](int base, int64_t a_rel, int64_t b_rel, int32_t lp_a, int32_t lp_b) {
      for (int32_t kk = lp_a; kk < lp_b; ++kk) {
        const int32_t li = lp_idx[kk];
        if ((uint32_t)li >= (uint32_t)n_rff) continue;
        const int64_t te = thr_enter[base * n_rff + li];
        const int64_t tl = thr_leave[base * n_rff + li];
        const int64_t s_high = te - a_rel - 1;
        int32_t i0 = 0, i1 = 0;
        if (b_rel == inf_ps) {
          if (s_high < starts0) continue;
          i0 = 0;
          i1 = (s_high >= startsN) ? n_st : upper_bound_i64(starts, n_st, s_high);
        } else {
          const int64_t s_low = tl - b_rel + 1;
          if (s_high < starts0 || s_low > startsN) continue;
          i0 = (s_low <= starts0) ? 0 : lower_bound_i64(starts, n_st, s_low);
          i1 = (s_high >= startsN) ? n_st : upper_bound_i64(starts, n_st, s_high);
        }
        if (i0 < i1) {
          const size_t off = (size_t)li * row_len;
          diffs[off + (size_t)i0] += 1;
          diffs[off + (size_t)i1] -= 1;
        }
      }
    };

    for (int32_t dpi = 0; dpi < n_dp; ++dpi) {
      const int32_t rec_slot = dp_slot[dpi];
      const int base = (dp_base[dpi] & 1);
      const int32_t lp_a = lp_ptr[dpi];
      const int32_t lp_b = lp_ptr[dpi + 1];
      if (lp_a >= lp_b) continue;
      if (rec_slot < 0 || rec_slot + 1 >= n_rec_ptr) continue;
      const int32_t a = eptr[rec_slot];
      const int32_t b = eptr[rec_slot + 1];
      if (a >= b) continue;
      for_each_pulse(et, ev, a, b, base, inf_ps,
                     [&](int64_t ta, int64_t tb) { process_interval(base, ta, tb, lp_a, lp_b); });
    }

    for (int32_t li = 0; li < n_rff; ++li) {
      int64_t acc = 0;
      const int32_t w = widx[li];
      if ((uint32_t)w >= (uint32_t)n_words) continue;
      const uint64_t m = wmsk[li];
      const size_t off = (size_t)li * row_len;
      for (int32_t i = 0; i < n_st; ++i) {
        acc += diffs[off + (size_t)i];
        if (acc > 0) outp[(size_t)i * (size_t)n_words + (size_t)w] |= m;
      }
    }
  }
  return out;
}

}  // namespace

PYBIND11_MODULE(_engine, m) {
  m.doc() = "setfi event-driven SET propagation and capture kernel";

  m.def("simulate_pulses", &simulate_pulses, py::arg("packed"), py::arg("delaytab"),
        py::arg("net_val_init_u8"), py::arg("src_i"), py::arg("v0"), py::arg("v1"),
        py::arg("pw_ps_array"), py::arg("sim_end_ps"), py::arg("eps_ps"), py::arg("enable_em"),
        py::arg("margin_ps"), py::arg("delay_policy_is_max"),
        R"doc(Inject one SET per pulse width on net src_i (flip v0 -> v1 at t=0,
back to v0 at t=pw) and propagate it through the packed cone.

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

  m.def("capture_masks", &capture_masks, py::arg("starts_ps"), py::arg("thr_enter_local"),
        py::arg("thr_leave_local"), py::arg("ev_ptr"), py::arg("ev_t"), py::arg("ev_v"),
        py::arg("dp_rec_slot"), py::arg("dp_base_u8"), py::arg("dp_localpos_ptr"),
        py::arg("dp_localpos_idx"), py::arg("word_index"), py::arg("word_mask"),
        py::arg("n_words"), py::arg("inf_ps"),
        R"doc(Capture bit masks on a grid of SET start times.

Returns uint64[n_starts, n_words]; bit li of row i is set when the local FF li
captures a wrong value for a SET starting at starts_ps[i].)doc");
}
