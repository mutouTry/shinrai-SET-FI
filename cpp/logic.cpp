// setfi._engine, part 4: the logic-only translation (setfi.campaign.logic).
//
//   logic_block  for every (cycle, site) of a block of cycles: invert the site in
//                the recorded state, re-evaluate in topological order the gates
//                of its cone whose inputs change, and list the observed FFs
//                whose D net changes.  No delays, no inertial filtering, no
//                capture window.
//
// The cones are those of run_block (setfi.campaign.fastpath), compiled without
// interconnect wires.  Their nets are numbered with the cone's topological list
// first, so a gate's output has a larger number than every input that can
// change: evaluating the queued gates in order of their output net sees each
// gate's inputs final, and evaluates it once.  A net's recorded value is read
// from the state when needed; only a cone with nets outside the recording is
// bound first (bind_state settles those nets).

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include "fastpath.hpp"

namespace py = pybind11;

namespace setfi_fast {
namespace {

// Scratch space and output rows of one thread.
struct LogicThread {
  std::vector<uint8_t> val, dp_base;  // the bound state, or only the changed values
  std::vector<int32_t> changed;       // per net: == stamp if its value changed
  std::vector<int32_t> queued;        // per gate: == stamp once it is queued
  std::vector<int32_t> heap;          // queued gates, the smallest output net on top
  std::vector<int32_t> slot_net;      // record slot -> net
  std::vector<int32_t> net_dp;        // net -> its index among the D nets (-1: not a D net)
  std::vector<int32_t> hit;           // D nets that changed
  std::vector<int32_t> upset;         // local FF indices of one (cycle, site)
  int32_t stamp = 0;
  std::vector<int32_t> cycle, site, ff;
  std::exception_ptr error;
  int64_t error_item = -1;
};

// Invert the site in the recorded state `row` (or in t.val if `bound`) and
// re-evaluate the gates that see a changed input; t.upset gets the local indices
// of the FFs whose D net changed, ascending.
void invert_site(const CompiledCone& c, const uint8_t* row, bool bound, LogicThread& t) {
  if (t.stamp == INT32_MAX - 1) {
    std::fill(t.changed.begin(), t.changed.end(), 0);
    std::fill(t.queued.begin(), t.queued.end(), 0);
    t.stamp = 0;
  }
  const int32_t st = ++t.stamp;
  auto later = [&](int32_t a, int32_t b) {  // heap order: gate a after gate b
    return c.gate_out[(size_t)a] != c.gate_out[(size_t)b] ? c.gate_out[(size_t)a] > c.gate_out[(size_t)b] : a > b;
  };
  auto value = [&](int32_t net) -> uint8_t {
    if (bound || t.changed[(size_t)net] == st) return t.val[(size_t)net] & 1;
    const int32_t g = c.gidx[(size_t)net];
    return g >= 0 ? (row[g] & 1) : 0;
  };
  int32_t now = -1;  // output net of the gate being evaluated
  auto set = [&](int32_t net, uint8_t v) {
    t.val[(size_t)net] = v;
    t.changed[(size_t)net] = st;
    if (t.net_dp[(size_t)net] >= 0) t.hit.push_back(t.net_dp[(size_t)net]);
    for (int32_t k = c.fanout_ptr[(size_t)net]; k < c.fanout_ptr[(size_t)net + 1]; k++) {
      const int32_t g = c.fanout_idx[(size_t)k];
      if (t.queued[(size_t)g] == st) continue;
      if (c.gate_out[(size_t)g] <= now) throw std::runtime_error("logic_block: a cone is not in topological order");
      t.queued[(size_t)g] = st;
      t.heap.push_back(g);
      std::push_heap(t.heap.begin(), t.heap.end(), later);
    }
  };
  t.heap.clear();
  t.hit.clear();
  set(c.src_i, (uint8_t)(value(c.src_i) ^ 1));
  while (!t.heap.empty()) {
    std::pop_heap(t.heap.begin(), t.heap.end(), later);
    const int32_t g = t.heap.back();
    t.heap.pop_back();
    now = c.gate_out[(size_t)g];
    int64_t idx = 0;
    int j = 0;
    for (int32_t q = c.gate_var_ptr[(size_t)g]; q < c.gate_var_ptr[(size_t)g + 1]; q++, j++) {
      if (j >= 62) throw std::out_of_range("logic_block: gate with too many inputs");
      idx |= (int64_t)value(c.gate_var_nets[(size_t)q]) << j;
    }
    const int32_t ta = c.gate_truth_ptr[(size_t)g], tb = c.gate_truth_ptr[(size_t)g + 1];
    if (tb - ta > 1 && idx >= (int64_t)(tb - ta)) throw std::out_of_range("logic_block: truth table index out of range");
    if (tb <= ta) throw std::out_of_range("logic_block: empty truth table");
    const uint8_t y = (uint8_t)(c.gate_truth[(size_t)(tb - ta > 1 ? ta + idx : ta)] & 1);
    if (y != value(now)) set(now, y);
  }
  t.upset.clear();
  for (int32_t dpi : t.hit)
    for (int32_t k = c.dp_ptr[(size_t)dpi]; k < c.dp_ptr[(size_t)dpi + 1]; k++) t.upset.push_back(c.dp_idx[(size_t)k]);
  std::sort(t.upset.begin(), t.upset.end());
  t.upset.erase(std::unique(t.upset.begin(), t.upset.end()), t.upset.end());
}

py::dict logic_block(py::list cones_list, py::array_t<uint8_t, py::array::c_style | py::array::forcecast> states,
                     py::array_t<int64_t, py::array::c_style | py::array::forcecast> cycles_arr,
                     py::array_t<int32_t, py::array::c_style | py::array::forcecast> site_ids_arr, int n_threads) {
  const int64_t n_sites = (int64_t)py::len(cones_list);
  std::vector<const CompiledCone*> cones((size_t)n_sites);
  for (int64_t j = 0; j < n_sites; j++) {
    cones[(size_t)j] = &py::cast<const CompiledCone&>(cones_list[(size_t)j]);
    if (!cones[(size_t)j]->wire_sink.empty())
      throw std::runtime_error("logic_block: cones must be compiled without interconnect wires");
  }
  if (states.ndim() != 2) throw std::runtime_error("logic_block: states must be [n_cycles, n_nets]");
  if (cycles_arr.ndim() != 1 || site_ids_arr.ndim() != 1)
    throw std::runtime_error("logic_block: cycles and site_ids must be 1-D");
  const int64_t n_cyc = (int64_t)cycles_arr.shape(0);
  if ((int64_t)states.shape(0) != n_cyc) throw std::runtime_error("logic_block: one state row per cycle");
  if ((int64_t)site_ids_arr.shape(0) != n_sites) throw std::runtime_error("logic_block: one site id per cone");
  const int64_t n_glob = (int64_t)states.shape(1);
  const uint8_t* states_p = states.data();
  const int64_t* cycles = cycles_arr.data();
  const int32_t* site_ids = site_ids_arr.data();
  for (int64_t k = 0; k < n_cyc; k++)
    if (cycles[k] < INT32_MIN || cycles[k] > INT32_MAX)
      throw std::overflow_error("logic_block: cycle " + std::to_string(cycles[k]) + " does not fit int32");

  int nt = n_threads < 1 ? 1 : n_threads;
  if ((int64_t)nt > n_sites) nt = (int)std::max<int64_t>(1, n_sites);
  // per item (cycle k, site j), k-major: where its rows are
  const size_t n_items = (size_t)(n_cyc * n_sites);
  std::vector<int64_t> row_start(n_items, 0), row_count(n_items, 0);
  std::vector<int32_t> thread_of((size_t)n_sites, 0);
  std::vector<LogicThread> outs((size_t)nt);
  // largest cones first, so that no thread is left with a big one at the end
  std::vector<int64_t> order((size_t)n_sites);
  for (int64_t j = 0; j < n_sites; j++) order[(size_t)j] = j;
  auto size_of = [&](int64_t j) {
    return (int64_t)cones[(size_t)j]->gate_var_nets.size() + (int64_t)cones[(size_t)j]->rec_idx.size();
  };
  std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) { return size_of(a) > size_of(b); });
  std::atomic<int64_t> next_site{0};
  std::atomic<bool> stop{false};
  bool interrupted = false;  // a Python signal handler raised (set by thread 0, the caller)
  SignalPoll poll;

  auto work = [&](int tid) {
    LogicThread& t = outs[(size_t)tid];
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
      const CompiledCone& c = *cones[(size_t)j];
      const size_t n = c.rec_idx.size();
      if (t.changed.size() < n) t.changed.resize(n, 0);
      if (t.val.size() < n) t.val.resize(n);
      if (t.queued.size() < c.gate_out.size()) t.queued.resize(c.gate_out.size(), 0);
      t.slot_net.assign(n, -1);
      for (size_t i = 0; i < n; i++)
        if (c.rec_idx[i] >= 0) t.slot_net[(size_t)c.rec_idx[i]] = (int32_t)i;
      t.net_dp.assign(n, -1);
      for (size_t dpi = 0; dpi < c.dp_rec_slot.size(); dpi++) {
        const int32_t slot = c.dp_rec_slot[dpi];
        if (slot >= 0 && (size_t)slot < n && t.slot_net[(size_t)slot] >= 0) t.net_dp[(size_t)t.slot_net[(size_t)slot]] = (int32_t)dpi;
      }
      const bool bound = !c.settle_gate.empty();
      const bool in_state = std::all_of(c.gidx.begin(), c.gidx.end(), [&](int32_t g) { return g < n_glob; });
      for (int64_t k = 0; k < n_cyc; k++) {
        const size_t item = (size_t)(k * n_sites + j);
        const uint8_t* row = states_p + (size_t)(k * n_glob);
        row_start[item] = (int64_t)t.ff.size();
        try {
          if (!in_state) throw std::out_of_range("global net index beyond the recorded state");
          if (bound) bind_state(c, row, n_glob, t.val, t.dp_base);
          invert_site(c, row, bound, t);
        } catch (...) {
          if (!t.error || (int64_t)item < t.error_item) {
            t.error = std::current_exception();
            t.error_item = (int64_t)item;
          }
          break;
        }
        for (int32_t li : t.upset) {
          t.cycle.push_back((int32_t)cycles[k]);
          t.site.push_back(site_ids[j]);
          t.ff.push_back(c.ffids[(size_t)li]);
        }
        row_count[item] = (int64_t)t.ff.size() - row_start[item];
      }
    }
  };
  {
    py::gil_scoped_release release;
    run_workers(nt, stop, work);
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

  // gather in k-major order
  int64_t n_rows = 0;
  for (size_t it = 0; it < n_items; it++) n_rows += row_count[it];
  py::array_t<int32_t> a_cycle((py::ssize_t)n_rows), a_site((py::ssize_t)n_rows), a_ff((py::ssize_t)n_rows);
  {
    int32_t* pc = a_cycle.mutable_data();
    int32_t* ps = a_site.mutable_data();
    int32_t* pf = a_ff.mutable_data();
    py::gil_scoped_release release;
    int64_t pos = 0;
    for (size_t it = 0; it < n_items; it++) {
      const LogicThread& t = outs[(size_t)thread_of[it % (size_t)n_sites]];
      const size_t a = (size_t)row_start[it], n = (size_t)row_count[it];
      std::copy(t.cycle.begin() + a, t.cycle.begin() + a + n, pc + pos);
      std::copy(t.site.begin() + a, t.site.begin() + a + n, ps + pos);
      std::copy(t.ff.begin() + a, t.ff.begin() + a + n, pf + pos);
      pos += (int64_t)n;
    }
  }
  py::dict r;
  r["cycle"] = a_cycle;
  r["site"] = a_site;
  r["ff"] = a_ff;
  return r;
}

}  // namespace

void register_logic(py::module_& m) {
  m.def("logic_block", &logic_block, py::arg("cones"), py::arg("states"), py::arg("cycles"), py::arg("site_ids"),
        py::arg("n_threads") = 1,
        R"doc(The logic-only translation of a block of cycles: for every cone (a CompiledCone
compiled without interconnect wires) and every cycle (row of states, one byte
per global net), invert the site in the recorded state, re-evaluate the gates
whose inputs change, and list the observed FFs whose D net changes.

Returns a dict of int32 columns cycle, site (from site_ids) and ff: one row
per FF upset, ordered by cycle (as given), then by cone, then by FF id.)doc");
}

}  // namespace setfi_fast
