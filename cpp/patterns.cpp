// setfi._engine.PatternBuilder: the pattern table of a campaign (see
// setfi.analysis.patterns), built from its pulse records one record file at a
// time.
//
// For a trial (cycle, site, width) and a grid of SET start times starts[k], F(k)
// is the set of FFs that capture a wrong value for a SET starting at starts[k].
// A record (ff, base, t_enter, t_exit) of the trial makes its FF capture for the
// start times s with
//     leave[base][ff] - t_exit + 1  <=  s  <=  enter[base][ff] - t_enter - 1
// (no lower bound when t_exit >= inf_ps; setfi.records), that is at the grid
// indices k0 <= k < k1.  enter and leave are the FF's window in the record's
// cycle: the FF's own, or one of the exceptions given by cycle.  With step_ps >=
// 0, a record with t_enter = 0 also captures at the grid point step_ps.
// Sweeping these range ends in order of k gives F(k),
// which only changes at a range end.  Each maximal range k_lo..k_hi of grid
// indices with one non-empty F is a run (site, width, k_lo, k_hi) of the
// pattern (cycle, F).
//
// add() sweeps the trials of one record file: the blocks of consecutive records
// with one (cycle, site, width).  setfi writes each trial as one block of one
// file.  A trial met in a second block (in another file, or elsewhere in the
// same file) is "split": its runs are wrong until forget_split_trials() drops
// them and add() is given all of its records at once.
//
// Patterns never span cycles, so each sampled cycle has a table of its own: its
// FF sets, their runs and the trials met so far.  add() may run on several
// files at once (from several Python threads); two calls only wait for each
// other while they merge runs into the same cycle's table.  result() puts the
// tables in a canonical order, which depends neither on the order of the files
// nor on how many ran at once.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "fastpath.hpp"

namespace py = pybind11;

namespace setfi_fast {
namespace {

using I32Arr = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;
using I64Arr = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;
using U8Arr = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;

// The columns of a dict of records (setfi.records.COLUMNS), as C arrays.
struct RecordColumns {
  I32Arr cycle_a, site_a, width_a, ff_a;
  U8Arr base_a;
  I64Arr t_enter_a, t_exit_a;
  const int32_t *cycle, *site, *width, *ff;
  const uint8_t* base;
  const int64_t *t_enter, *t_exit;
  int64_t n = 0;

  explicit RecordColumns(const py::dict& cols) {
    cycle_a = get<I32Arr>(cols, "cycle");
    site_a = get<I32Arr>(cols, "site");
    width_a = get<I32Arr>(cols, "width_ps");
    ff_a = get<I32Arr>(cols, "ff");
    base_a = get<U8Arr>(cols, "base");
    t_enter_a = get<I64Arr>(cols, "t_enter");
    t_exit_a = get<I64Arr>(cols, "t_exit");
    n = (int64_t)cycle_a.size();
    for (int64_t m : {(int64_t)site_a.size(), (int64_t)width_a.size(), (int64_t)ff_a.size(),
                      (int64_t)base_a.size(), (int64_t)t_enter_a.size(), (int64_t)t_exit_a.size()})
      if (m != n) throw std::invalid_argument("records: the columns differ in length");
    cycle = cycle_a.data();
    site = site_a.data();
    width = width_a.data();
    ff = ff_a.data();
    base = base_a.data();
    t_enter = t_enter_a.data();
    t_exit = t_exit_a.data();
  }

  bool same_trial(int64_t i, int64_t j) const {
    return cycle[i] == cycle[j] && site[i] == site[j] && width[i] == width[j];
  }

 private:
  template <typename Arr>
  static Arr get(const py::dict& cols, const char* name) {
    if (!cols.contains(name)) throw std::invalid_argument(std::string("records: no column '") + name + "'");
    Arr a = py::cast<Arr>(cols[name]);
    if (a.ndim() != 1) throw std::invalid_argument(std::string("records: column '") + name + "' is not 1-D");
    return a;
  }
};

// One block of consecutive records of one trial.  key = site * n_widths + width
// index numbers the trial within its cycle.
struct Trial {
  int64_t begin, end, key;
  int32_t cycle;  // index into the sampled cycles
  int32_t site, width_ps;
};

// Grid indices k_lo..k_hi (inclusive) of one trial, all with the FF set
// `pattern` (a number in the cycle's FFSets).
struct Run {
  int32_t pattern, site, width_ps, k_lo, k_hi;
};

// One end of the grid range of a record: FF ff enters (+1) or leaves (-1) at k.
struct RangeEnd {
  int32_t k, ff, delta;
  bool operator<(const RangeEnd& o) const { return k != o.k ? k < o.k : ff < o.ff; }
};

// The hash of an FF set is the XOR of its FFs' hashes, so that it follows the
// set as FFs enter and leave.
inline uint64_t ff_hash(int32_t ff) {
  uint64_t x = (uint64_t)(uint32_t)ff + 0x9e3779b97f4a7c15ULL;  // the splitmix64 finalizer
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

// Distinct FF sets (ascending FF ids), numbered in order of arrival.
class FFSets {
 public:
  int32_t size() const { return (int32_t)hash_.size(); }
  uint64_t hash(int32_t p) const { return hash_[(size_t)p]; }
  const int32_t* ffs(int32_t p) const { return ffs_.data() + ptr_[(size_t)p]; }
  int32_t n_ffs(int32_t p) const { return (int32_t)(ptr_[(size_t)p + 1] - ptr_[(size_t)p]); }

  // The number of the set ffs[0 .. n) with hash h; a new set gets the next number.
  int32_t intern(const int32_t* ffs, int32_t n, uint64_t h) {
    if (2 * ((size_t)size() + 1) > slots_.size()) grow();
    const size_t mask = slots_.size() - 1;
    for (size_t i = (size_t)h & mask;; i = (i + 1) & mask) {
      const int32_t p = slots_[i] - 1;
      if (p < 0) {
        if (size() == INT32_MAX) throw std::length_error("PatternBuilder: more than 2^31 - 1 patterns in a cycle");
        slots_[i] = size() + 1;
        hash_.push_back(h);
        ffs_.insert(ffs_.end(), ffs, ffs + n);
        ptr_.push_back((int64_t)ffs_.size());
        return size() - 1;
      }
      if (hash_[(size_t)p] == h && n_ffs(p) == n && std::equal(ffs, ffs + n, this->ffs(p))) return p;
    }
  }

  void clear() {
    hash_.clear();
    ffs_.clear();
    ptr_.assign(1, 0);
    std::fill(slots_.begin(), slots_.end(), 0);
  }

 private:
  // twice the slots, at most half of them used (linear probing)
  void grow() {
    slots_.assign(std::max<size_t>(64, 2 * slots_.size()), 0);
    const size_t mask = slots_.size() - 1;
    for (int32_t p = 0; p < size(); p++) {
      size_t i = (size_t)hash_[(size_t)p] & mask;
      while (slots_[i] != 0) i = (i + 1) & mask;
      slots_[i] = p + 1;
    }
  }

  std::vector<uint64_t> hash_;
  std::vector<int64_t> ptr_{0};  // FF ids of set p: ffs_[ptr_[p] .. ptr_[p + 1])
  std::vector<int32_t> ffs_;
  std::vector<int32_t> slots_;   // set number + 1; 0 = empty
};

// What is known about one sampled cycle.
struct CycleTable {
  std::mutex lock;
  FFSets sets;                          // its patterns' FF sets
  std::vector<Run> runs;
  std::vector<uint64_t> seen;           // bit per trial key: met in a block of records
  std::unordered_set<int64_t> split;    // trial keys met in more than one block
};

// The runs of some trials of one cycle before they go to the cycle's table,
// and scratch space.
struct Sweep {
  FFSets sets;
  std::vector<Run> runs;                        // Run::pattern numbers `sets`
  std::vector<RangeEnd> ends;
  std::vector<int32_t> count;                   // per FF id: records covering the current k
  std::vector<int32_t> members, toggled, next;  // F(k), FFs entering or leaving at k, F(k) after
  std::vector<int32_t> number;                  // set in `sets` -> set in the cycle's table
};

class PatternBuilder {
 public:
  PatternBuilder(I64Arr starts_ps, I64Arr enter, I64Arr leave, I64Arr cycles, int64_t n_sites,
                 I64Arr widths_ps, int64_t inf_ps, I64Arr exc_cycles, I64Arr exc_ffs, I64Arr exc_enter,
                 I64Arr exc_leave, int64_t step_ps)
      : n_sites_(n_sites), inf_ps_(inf_ps) {
    if (starts_ps.ndim() != 1 || starts_ps.size() == 0)
      throw std::invalid_argument("PatternBuilder: starts_ps must be a non-empty 1-D array");
    starts_.assign(starts_ps.data(), starts_ps.data() + starts_ps.size());
    if (!std::is_sorted(starts_.begin(), starts_.end()) || starts_.size() > (size_t)INT32_MAX - 1)
      throw std::invalid_argument("PatternBuilder: starts_ps must be ascending");
    if (enter.ndim() != 2 || leave.ndim() != 2 || enter.shape(0) != 2 || leave.shape(0) != 2 ||
        enter.shape(1) != leave.shape(1))
      throw std::invalid_argument("PatternBuilder: enter and leave must be [2, n_ff_ids]");
    n_ff_ = (int32_t)enter.shape(1);
    enter_.assign(enter.data(), enter.data() + enter.size());
    leave_.assign(leave.data(), leave.data() + leave.size());
    if (cycles.ndim() != 1 || widths_ps.ndim() != 1 || n_sites < 0)
      throw std::invalid_argument("PatternBuilder: cycles and widths_ps must be 1-D, n_sites >= 0");
    for (int64_t i = 0; i < (int64_t)cycles.size(); i++) {
      cycles_.push_back((int32_t)cycles.data()[i]);
      cycle_index_.emplace(cycles_.back(), (int32_t)i);
      tables_.push_back(std::make_unique<CycleTable>());
    }
    for (int64_t i = 0; i < (int64_t)widths_ps.size(); i++) width_index_.emplace((int32_t)widths_ps.data()[i], i);
    n_widths_ = (int64_t)widths_ps.size();
    if ((double)n_sites * (double)n_widths_ > 1e11)
      throw std::invalid_argument("PatternBuilder: more than 10^11 trials per cycle");
    const int64_t n_exc = (int64_t)exc_cycles.size();
    if (exc_ffs.size() != n_exc || exc_enter.size() != 2 * n_exc || exc_leave.size() != 2 * n_exc)
      throw std::invalid_argument("PatternBuilder: exc_ffs, exc_enter [n, 2], exc_leave [n, 2] must match exc_cycles");
    exc_enter_.assign(exc_enter.data(), exc_enter.data() + 2 * n_exc);
    exc_leave_.assign(exc_leave.data(), exc_leave.data() + 2 * n_exc);
    for (int64_t i = 0; i < n_exc; i++) {
      const int64_t f = exc_ffs.data()[i];
      if (f < 0 || f >= n_ff_) throw std::invalid_argument("PatternBuilder: exception FF id out of range");
      exc_row_[exc_key(cycle_index((int32_t)exc_cycles.data()[i]), (int32_t)f)] = i;
    }
    step_k_ = -1;
    if (step_ps >= 0) {
      const auto it = std::lower_bound(starts_.begin(), starts_.end(), step_ps);
      if (it != starts_.end() && *it == step_ps) step_k_ = (int32_t)(it - starts_.begin());
    }
  }

  // Sweep the trials of one dict of records (a record file) into the tables.
  void add(const py::dict& cols) {
    const RecordColumns rc(cols);
    py::gil_scoped_release release;
    const std::vector<Trial> trials = find_trials(rc);
    Sweep o;
    o.count.assign((size_t)n_ff_, 0);
    // a cycle's trials at a time (setfi writes the cycles of a file one after the other)
    for (size_t a = 0, b = 0; a < trials.size(); a = b) {
      while (b < trials.size() && trials[b].cycle == trials[a].cycle) b++;
      for (size_t t = a; t < b; t++) sweep_trial(rc, trials[t], o);
      merge(trials, a, b, o);
    }
  }

  int64_t n_split_trials() const {
    int64_t n = 0;
    for (const auto& c : tables_) n += (int64_t)c->split.size();
    return n;
  }

  // Which of these records belong to a split trial.
  py::array_t<bool> split_mask(const py::dict& cols) const {
    const RecordColumns rc(cols);
    py::array_t<bool> out((py::ssize_t)rc.n);
    bool* m = out.mutable_data();
    for (int64_t i = 0; i < rc.n; i++) {
      const CycleTable& c = *tables_[(size_t)cycle_index(rc.cycle[i])];
      m[i] = c.split.count(trial_key(rc.site[i], rc.width[i])) > 0;
    }
    return out;
  }

  // Drop the runs of the split trials and forget that they were met: add()
  // takes all their records next, at once.
  void forget_split_trials() {
    for (auto& cp : tables_) {
      CycleTable& c = *cp;
      if (c.split.empty()) continue;
      c.runs.erase(std::remove_if(c.runs.begin(), c.runs.end(),
                                  [&](const Run& r) { return c.split.count(trial_key(r.site, r.width_ps)) > 0; }),
                   c.runs.end());
      for (int64_t key : c.split) c.seen[(size_t)(key >> 6)] &= ~(1ULL << (key & 63));
      c.split.clear();
    }
  }

  // The table: patterns sorted by (cycle, FF ids), the runs of a pattern by
  // (site, width_ps, k_lo).  FF sets left without runs by forget_split_trials()
  // are not patterns.  Each cycle's table is freed once written, so the
  // builder is empty afterwards.
  py::dict result() {
    if (n_split_trials() > 0)
      throw std::runtime_error("PatternBuilder.result: split trials are pending (see forget_split_trials)");
    // the cycles in ascending order; in each, the FF sets that have runs, ascending
    std::vector<int32_t> cycle_order(cycles_.size());
    std::iota(cycle_order.begin(), cycle_order.end(), 0);
    std::sort(cycle_order.begin(), cycle_order.end(),
              [&](int32_t a, int32_t b) { return cycles_[(size_t)a] < cycles_[(size_t)b]; });
    std::vector<std::vector<int32_t>> order(cycles_.size());
    std::vector<std::vector<int64_t>> n_runs(cycles_.size());
    int64_t n_pat = 0, n_ff_total = 0, n_run = 0;
    for (int32_t ci : cycle_order) {
      const CycleTable& c = *tables_[(size_t)ci];
      std::vector<int64_t>& nr = n_runs[(size_t)ci];
      nr.assign((size_t)c.sets.size(), 0);
      for (const Run& r : c.runs) nr[(size_t)r.pattern]++;
      std::vector<int32_t>& o = order[(size_t)ci];
      for (int32_t p = 0; p < c.sets.size(); p++)
        if (nr[(size_t)p] > 0) o.push_back(p);
      std::sort(o.begin(), o.end(), [&](int32_t a, int32_t b) {
        return std::lexicographical_compare(c.sets.ffs(a), c.sets.ffs(a) + c.sets.n_ffs(a), c.sets.ffs(b),
                                            c.sets.ffs(b) + c.sets.n_ffs(b));
      });
      n_pat += (int64_t)o.size();
      for (int32_t p : o) n_ff_total += c.sets.n_ffs(p);
      n_run += (int64_t)c.runs.size();
    }

    py::array_t<int32_t> cycle((py::ssize_t)n_pat), ffs((py::ssize_t)n_ff_total), n_sites((py::ssize_t)n_pat);
    py::array_t<int64_t> ff_ptr((py::ssize_t)n_pat + 1), multiplicity((py::ssize_t)n_pat),
        run_ptr((py::ssize_t)n_pat + 1);
    py::array_t<int32_t> run_site((py::ssize_t)n_run), run_width((py::ssize_t)n_run), run_k_lo((py::ssize_t)n_run),
        run_k_hi((py::ssize_t)n_run);
    Columns out{cycle.mutable_data(), ffs.mutable_data(), n_sites.mutable_data(), ff_ptr.mutable_data(),
                multiplicity.mutable_data(), run_ptr.mutable_data(), run_site.mutable_data(),
                run_width.mutable_data(), run_k_lo.mutable_data(), run_k_hi.mutable_data()};
    {
      py::gil_scoped_release release;
      out.ff_ptr[0] = 0;
      out.run_ptr[0] = 0;
      int64_t i = 0;
      for (int32_t ci : cycle_order) {
        write_cycle(ci, order[(size_t)ci], n_runs[(size_t)ci], i, out);
        i += (int64_t)order[(size_t)ci].size();
        tables_[(size_t)ci] = std::make_unique<CycleTable>();
      }
    }
    py::dict d;
    d["pattern_cycle"] = cycle;
    d["pattern_ff_ptr"] = ff_ptr;
    d["pattern_ffs"] = ffs;
    d["pattern_multiplicity"] = multiplicity;
    d["pattern_n_sites"] = n_sites;
    d["pattern_run_ptr"] = run_ptr;
    d["run_site"] = run_site;
    d["run_width_ps"] = run_width;
    d["run_k_lo"] = run_k_lo;
    d["run_k_hi"] = run_k_hi;
    return d;
  }

 private:
  // the output arrays of result()
  struct Columns {
    int32_t *cycle, *ffs, *n_sites;
    int64_t *ff_ptr, *multiplicity, *run_ptr;
    int32_t *run_site, *run_width, *run_k_lo, *run_k_hi;
  };

  int32_t cycle_index(int32_t cycle) const {
    const auto c = cycle_index_.find(cycle);
    if (c == cycle_index_.end())
      throw std::invalid_argument("records: cycle " + std::to_string(cycle) + " is not a sampled cycle");
    return c->second;
  }

  int64_t exc_key(int32_t cycle_idx, int32_t ff) const { return (int64_t)cycle_idx * n_ff_ + ff; }

  int64_t trial_key(int32_t site, int32_t width_ps) const {
    const auto w = width_index_.find(width_ps);
    if (w == width_index_.end())
      throw std::invalid_argument("records: width " + std::to_string(width_ps) + " ps was not injected");
    if (site < 0 || site >= n_sites_)
      throw std::invalid_argument("records: site " + std::to_string(site) + " is not a site index");
    return (int64_t)site * n_widths_ + w->second;
  }

  // The blocks of records with one (cycle, site, width), in file order; checks
  // the records.
  std::vector<Trial> find_trials(const RecordColumns& rc) const {
    std::vector<Trial> trials;
    int32_t ci = -1;
    for (int64_t i = 0, j = 0; i < rc.n; i = j) {
      for (; j < rc.n && rc.same_trial(i, j); j++)
        if (rc.ff[j] < 0 || rc.ff[j] >= n_ff_ || rc.base[j] > 1)
          throw std::invalid_argument("records: FF id " + std::to_string(rc.ff[j]) + " or base value " +
                                      std::to_string((int)rc.base[j]) + " out of range");
      if (ci < 0 || rc.cycle[i] != cycles_[(size_t)ci]) ci = cycle_index(rc.cycle[i]);
      trials.push_back({i, j, trial_key(rc.site[i], rc.width[i]), ci, rc.site[i], rc.width[i]});
    }
    return trials;
  }

  // The runs of one trial (records t.begin .. t.end), appended to o.runs.
  void sweep_trial(const RecordColumns& rc, const Trial& t, Sweep& o) const {
    const int64_t* s = starts_.data();
    const int32_t n_st = (int32_t)starts_.size();
    o.ends.clear();
    for (int64_t i = t.begin; i < t.end; i++) {
      const size_t at = (size_t)rc.base[i] * (size_t)n_ff_ + (size_t)rc.ff[i];
      int64_t enter = enter_[at], leave = leave_[at];
      if (!exc_row_.empty()) {
        const auto e = exc_row_.find(exc_key(t.cycle, rc.ff[i]));
        if (e != exc_row_.end()) {
          enter = exc_enter_[(size_t)(2 * e->second + rc.base[i])];
          leave = exc_leave_[(size_t)(2 * e->second + rc.base[i])];
        }
      }
      const int64_t s_high = enter - rc.t_enter[i] - 1;
      const int32_t k1 = (int32_t)(std::upper_bound(s, s + n_st, s_high) - s);
      const int32_t k0 = rc.t_exit[i] >= inf_ps_
                             ? 0
                             : (int32_t)(std::lower_bound(s, s + n_st, leave - rc.t_exit[i] + 1) - s);
      if (k0 < k1) {
        o.ends.push_back({k0, rc.ff[i], +1});
        o.ends.push_back({k1, rc.ff[i], -1});
      }
      if (step_k_ >= 0 && rc.t_enter[i] == 0 && !(k0 <= step_k_ && step_k_ < k1)) {
        o.ends.push_back({step_k_, rc.ff[i], +1});
        o.ends.push_back({step_k_ + 1, rc.ff[i], -1});
      }
    }
    std::sort(o.ends.begin(), o.ends.end());
    o.members.clear();
    uint64_t h = 0;  // the hash of members
    int32_t k_lo = 0;
    for (size_t i = 0; i < o.ends.size();) {
      // every range end at k, one FF at a time
      const int32_t k = o.ends[i].k;
      o.toggled.clear();
      while (i < o.ends.size() && o.ends[i].k == k) {
        const int32_t f = o.ends[i].ff;
        const bool was_in = o.count[(size_t)f] > 0;
        for (; i < o.ends.size() && o.ends[i].k == k && o.ends[i].ff == f; i++)
          o.count[(size_t)f] += o.ends[i].delta;
        if (was_in != (o.count[(size_t)f] > 0)) o.toggled.push_back(f);
      }
      if (o.toggled.empty()) continue;  // F(k) = F(k - 1): the run goes on
      if (!o.members.empty())
        o.runs.push_back(
            {o.sets.intern(o.members.data(), (int32_t)o.members.size(), h), t.site, t.width_ps, k_lo, k - 1});
      // F(k) = F(k - 1) with the toggled FFs added or removed (both ascending)
      o.next.clear();
      std::set_symmetric_difference(o.members.begin(), o.members.end(), o.toggled.begin(), o.toggled.end(),
                                    std::back_inserter(o.next));
      o.members.swap(o.next);
      for (int32_t f : o.toggled) h ^= ff_hash(f);
      k_lo = k;
    }
  }

  // Move the runs of trials[a .. b) (all of one cycle) from o to the cycle's
  // table, and note the trials met before (split trials).
  void merge(const std::vector<Trial>& trials, size_t a, size_t b, Sweep& o) {
    CycleTable& c = *tables_[(size_t)trials[a].cycle];
    std::lock_guard<std::mutex> hold(c.lock);
    if (c.seen.empty()) c.seen.assign(((size_t)(n_sites_ * n_widths_) + 63) / 64, 0);
    for (size_t t = a; t < b; t++) {
      uint64_t& word = c.seen[(size_t)(trials[t].key >> 6)];
      const uint64_t bit = 1ULL << (trials[t].key & 63);
      if (word & bit) c.split.insert(trials[t].key);
      word |= bit;
    }
    o.number.resize((size_t)o.sets.size());
    for (int32_t p = 0; p < o.sets.size(); p++)
      o.number[(size_t)p] = c.sets.intern(o.sets.ffs(p), o.sets.n_ffs(p), o.sets.hash(p));
    for (Run r : o.runs) {
      r.pattern = o.number[(size_t)r.pattern];
      c.runs.push_back(r);
    }
    o.sets.clear();
    o.runs.clear();
  }

  // The patterns of cycle ci (its FF sets `order`), numbered from i, and their
  // runs, into the output arrays.
  void write_cycle(int32_t ci, const std::vector<int32_t>& order, const std::vector<int64_t>& n_runs, int64_t i,
                   const Columns& out) const {
    const CycleTable& c = *tables_[(size_t)ci];
    std::vector<int64_t> next(c.sets.size(), -1);  // FF set -> the next free row of its runs
    for (int64_t q = 0; q < (int64_t)order.size(); q++) {
      const int32_t p = order[(size_t)q];
      out.cycle[i + q] = cycles_[(size_t)ci];
      std::copy(c.sets.ffs(p), c.sets.ffs(p) + c.sets.n_ffs(p), out.ffs + out.ff_ptr[i + q]);
      out.ff_ptr[i + q + 1] = out.ff_ptr[i + q] + c.sets.n_ffs(p);
      out.run_ptr[i + q + 1] = out.run_ptr[i + q] + n_runs[(size_t)p];
      next[(size_t)p] = out.run_ptr[i + q];
    }
    // the runs of each pattern, first in order of arrival ...
    for (const Run& r : c.runs) {
      const int64_t j = next[(size_t)r.pattern]++;
      out.run_site[j] = r.site;
      out.run_width[j] = r.width_ps;
      out.run_k_lo[j] = r.k_lo;
      out.run_k_hi[j] = r.k_hi;
    }
    // ... then sorted by (site, width_ps, k_lo)
    auto by_place = [](const Run& a, const Run& b) {
      if (a.site != b.site) return a.site < b.site;
      return a.width_ps != b.width_ps ? a.width_ps < b.width_ps : a.k_lo < b.k_lo;
    };
    std::vector<Run> rows;
    for (int64_t q = i; q < i + (int64_t)order.size(); q++) {
      rows.clear();
      for (int64_t j = out.run_ptr[q]; j < out.run_ptr[q + 1]; j++)
        rows.push_back({0, out.run_site[j], out.run_width[j], out.run_k_lo[j], out.run_k_hi[j]});
      if (!std::is_sorted(rows.begin(), rows.end(), by_place)) std::sort(rows.begin(), rows.end(), by_place);
      int64_t j = out.run_ptr[q], m = 0;
      int32_t sites = 0;
      for (size_t x = 0; x < rows.size(); x++, j++) {
        const Run& r = rows[x];
        out.run_site[j] = r.site;
        out.run_width[j] = r.width_ps;
        out.run_k_lo[j] = r.k_lo;
        out.run_k_hi[j] = r.k_hi;
        m += (int64_t)r.k_hi - r.k_lo + 1;
        sites += (x == 0 || r.site != rows[x - 1].site) ? 1 : 0;
      }
      out.multiplicity[q] = m;
      out.n_sites[q] = sites;
    }
  }

  std::vector<int64_t> starts_, enter_, leave_;  // enter_ / leave_: [2, n_ff_] row-major
  std::vector<int64_t> exc_enter_, exc_leave_;    // [n_exc, 2] row-major
  std::unordered_map<int64_t, int64_t> exc_row_;  // (cycle index, ff) -> exception row
  int32_t step_k_ = -1;                           // the grid index of step_ps, or -1
  int32_t n_ff_ = 0;
  int64_t n_sites_ = 0, n_widths_ = 0;
  int64_t inf_ps_ = 0;
  std::vector<int32_t> cycles_;
  std::unordered_map<int32_t, int32_t> cycle_index_;
  std::unordered_map<int32_t, int64_t> width_index_;
  std::vector<std::unique_ptr<CycleTable>> tables_;  // by cycle index
};

}  // namespace

void register_patterns(py::module_& m) {
  py::class_<PatternBuilder>(m, "PatternBuilder", R"doc(The pattern table of a campaign, built one
record file at a time (setfi.analysis.patterns.compute_patterns drives it).

starts_ps: the grid of SET start times; enter, leave: the capture thresholds
[2, n_ff_ids] (setfi.records.Records.thresholds); cycles, n_sites, widths_ps:
the trials of the campaign; inf_ps: the t_exit of an open-ended pulse;
exc_cycles, exc_ffs, exc_enter, exc_leave: the windows that differ by cycle
(setfi.records.Records.window_exceptions); step_ps: with propagation.vcs_align,
the capture edge, at which a record with t_enter = 0 captures (-1: none).)doc")
      .def(py::init<I64Arr, I64Arr, I64Arr, I64Arr, int64_t, I64Arr, int64_t, I64Arr, I64Arr, I64Arr, I64Arr,
                    int64_t>(),
           py::arg("starts_ps"), py::arg("enter"), py::arg("leave"), py::arg("cycles"), py::arg("n_sites"),
           py::arg("widths_ps"), py::arg("inf_ps"), py::arg("exc_cycles") = I64Arr(0),
           py::arg("exc_ffs") = I64Arr(0), py::arg("exc_enter") = I64Arr(0), py::arg("exc_leave") = I64Arr(0),
           py::arg("step_ps") = -1)
      .def("add", &PatternBuilder::add, py::arg("records"),
           R"doc(Sweep the trials of a dict of record columns (one record file): each block of
consecutive records with one (cycle, site, width_ps) is one trial.  Several
threads may call add() at once.)doc")
      .def_property_readonly("n_split_trials", &PatternBuilder::n_split_trials,
                             "Trials met in more than one block of records so far.")
      .def("split_mask", &PatternBuilder::split_mask, py::arg("records"),
           "Boolean array: which of these records belong to a split trial.")
      .def("forget_split_trials", &PatternBuilder::forget_split_trials,
           R"doc(Drop the runs of the split trials; add() must then get all their records at once,
sorted by (cycle, site, width_ps).)doc")
      .def("result", &PatternBuilder::result,
           R"doc(The table as a dict of arrays (the pattern_* and run_* arrays of
setfi.analysis.patterns.Patterns), in canonical order.  Empties the builder.)doc");
}

}  // namespace setfi_fast
