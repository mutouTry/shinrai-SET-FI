// setfi._engine: data and helpers shared by the trial loop (engine.cpp,
// run_block), the cone compiler (cone_compiler.cpp) and the pattern table
// (patterns.cpp).
#pragma once

#include <pybind11/pybind11.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

namespace setfi_fast {

// Runs work(0) on the calling thread and work(1) .. work(n - 1) on new threads,
// then joins them.  work must not throw, and must return soon once `stop` is
// set; if a thread cannot be started, the started ones are stopped and the
// error is rethrown.
template <typename Work>
inline void run_workers(int n, std::atomic<bool>& stop, Work& work) {
  std::vector<std::thread> threads;
  threads.reserve((size_t)(n > 1 ? n - 1 : 0));
  try {
    for (int t = 1; t < n; t++) threads.emplace_back([&work, t] { work(t); });
  } catch (...) {
    stop.store(true);
    for (auto& th : threads) th.join();
    throw;
  }
  work(0);
  for (auto& th : threads) th.join();
}

// For the calling thread of a loop that runs without the GIL: pending() takes
// the GIL at most every 50 ms and runs the Python signal handlers, so that
// Ctrl-C does not wait for the loop to end.  It returns true when a handler
// raised (KeyboardInterrupt); that exception stays set, and the caller throws
// pybind11::error_already_set() once its workers have stopped and it holds
// the GIL again.
class SignalPoll {
 public:
  bool pending() {
    const auto now = std::chrono::steady_clock::now();
    if (now < next_) return false;
    next_ = now + std::chrono::milliseconds(50);
    pybind11::gil_scoped_acquire gil;
    return PyErr_CheckSignals() != 0;
  }

 private:
  std::chrono::steady_clock::time_point next_ = std::chrono::steady_clock::now();
};

// One injection site, compiled for the trial loop: the packed cone
// (setfi.campaign.cone.PackedCone), its capture plan (CapturePlan), and what is
// needed to bind a recorded state to it (gidx, settle, wire pairs, dp_g; see
// setfi.campaign.run).  Every array has the dtype and content of its Python
// counterpart; CompiledCone.arrays() returns them under the same names.
struct CompiledCone {
  // PackedCone
  std::vector<int32_t> fanout_ptr, fanout_idx, gate_out, gate_var_ptr, gate_var_nets, gate_truth_ptr;
  std::vector<uint8_t> gate_truth;
  std::vector<int32_t> arc_dkey_id, arc_cause_i;
  std::vector<int8_t> arc_side_mode;
  std::vector<int32_t> arc_side_ptr, arc_side_nets, rec_idx;
  // transport wires; all empty when the cone has none (wire_ptr / wire_zsplit
  // are then all zero in PackedCone and never read by the engine)
  std::vector<int32_t> wire_ptr, wire_zsplit, wire_sink;
  std::vector<int64_t> wire_rise_ps, wire_fall_ps;

  // CapturePlan
  std::vector<int32_t> ffids, dp_rec_slot, dp_ptr, dp_idx;

  // binding a recorded state
  int32_t src_i = -1;
  std::vector<int32_t> gidx;                                   // local net -> global net (-1: not recorded)
  std::vector<int32_t> settle_gate, settle_in_ptr, settle_in;  // gates whose output is not recorded
  std::vector<int32_t> wp_drv, wp_sink;                        // wire pairs, in creation order
  std::vector<int64_t> dp_g;                                   // global net of each recorded D net

  // name ids (ConeCompiler numbering) of the D nets and, if requested, of every
  // local net (-1: a wire pin node); only for inspection
  std::vector<int32_t> dp_nid, net_nid;

  // ConeStats contributions of this site
  int64_t unconnected_side_pins = 0;
  int64_t dnets_without_observed_ff = 0;

  int32_t n_nets() const { return (int32_t)rec_idx.size(); }
  int32_t n_gates() const { return (int32_t)gate_out.size(); }
  int32_t n_ff() const { return (int32_t)ffids.size(); }
};

// Registers CompiledCone, ConeCompiler and parse_site_cones (cone_compiler.cpp).
void register_cone_compiler(pybind11::module_& m);

// Registers PatternBuilder (patterns.cpp).
void register_patterns(pybind11::module_& m);

}  // namespace setfi_fast
