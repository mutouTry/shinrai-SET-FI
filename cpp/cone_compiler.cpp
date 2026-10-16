// setfi._engine, part 2: loading site_cones.jsonl and compiling per-site cones.
//
//   parse_site_cones  the reader of substrate.load_skeleton for the index form
//                     written by the build, into CSR arrays.  Returns None for
//                     anything else (name form, floats, escapes, CR, a file it
//                     cannot read, ...), and the caller then uses the Python
//                     reader.
//   ConeCompiler      cone.build_cone (with insert_transport_nodes), pack_cone,
//                     build_capture_plan and the state-binding data of
//                     campaign.run, on integer tables prepared by
//                     setfi.campaign.fastpath.  The result is the same arrays the
//                     Python code produces.  A site the Python code would reject
//                     gets a non-zero status, and the caller runs the Python code
//                     for it (which raises the error).
//   CompiledCone      the compiled site (fastpath.hpp), consumed by run_block.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "fastpath.hpp"

namespace py = pybind11;

namespace setfi_fast {
namespace {

using I32A = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;
using I64A = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;
using U8A = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;
using I8A = py::array_t<int8_t, py::array::c_style | py::array::forcecast>;
using U64A = py::array_t<uint64_t, py::array::c_style | py::array::forcecast>;

template <typename T>
py::array_t<T> to_array(const std::vector<T>& v) {
  py::array_t<T> a((py::ssize_t)v.size());
  if (!v.empty()) std::memcpy(a.mutable_data(), v.data(), v.size() * sizeof(T));
  return a;
}

// a numpy array that owns the vector's buffer (no copy)
template <typename T>
py::array_t<T> move_to_array(std::vector<T>&& v) {
  if (v.empty()) return py::array_t<T>((py::ssize_t)0);
  auto* p = new std::vector<T>(std::move(v));
  py::capsule owner(p, [](void* x) { delete reinterpret_cast<std::vector<T>*>(x); });
  return py::array_t<T>({(py::ssize_t)p->size()}, {(py::ssize_t)sizeof(T)}, p->data(), owner);
}

template <typename T>
std::vector<T> from_array(const py::dict& d, const char* key) {
  if (!d.contains(key)) throw std::runtime_error(std::string("CompiledCone.from_arrays: missing '") + key + "'");
  auto a = py::cast<py::array_t<T, py::array::c_style | py::array::forcecast>>(d[key]);
  const T* p = a.data();
  return std::vector<T>(p, p + a.size());
}

// ---------------------------------------------------------------------------
// parse_site_cones
// ---------------------------------------------------------------------------
struct SkelOut {
  std::vector<int64_t> src;
  std::vector<int64_t> topo_ptr{0};
  std::vector<int32_t> topo;
  std::vector<int64_t> adj_ptr{0};
  std::vector<int32_t> adj_u;
  std::vector<int64_t> adj_eptr{0};
  std::vector<int32_t> eids;
  std::vector<int64_t> dn_ptr{0};
  std::vector<int32_t> dn;
  std::vector<int8_t> reconv;  // has_reconv: -1 absent / null, 0 false, 1 true
};

// A strict JSON reader for one line.  Every method returns false on anything it
// does not handle; the whole file then goes to the Python reader.
struct LineParser {
  const char* p;
  const char* end;

  void ws() {
    while (p < end && (*p == ' ' || *p == '\t')) ++p;
  }
  bool lit(char c) {
    ws();
    if (p < end && *p == c) {
      ++p;
      return true;
    }
    return false;
  }
  bool keyword(const char* kw) {
    ws();
    const size_t n = std::strlen(kw);
    if ((size_t)(end - p) >= n && std::memcmp(p, kw, n) == 0) {
      p += n;
      return true;
    }
    return false;
  }
  // a JSON integer: no fraction or exponent, at most 18 digits
  bool integer(int64_t& v) {
    ws();
    bool neg = false;
    if (p < end && *p == '-') {
      neg = true;
      ++p;
    }
    if (p >= end || *p < '0' || *p > '9') return false;
    if (*p == '0' && p + 1 < end && p[1] >= '0' && p[1] <= '9') return false;
    int64_t x = 0;
    int nd = 0;
    while (p < end && *p >= '0' && *p <= '9') {
      x = x * 10 + (*p - '0');
      ++p;
      if (++nd > 18) return false;
    }
    if (p < end && (*p == '.' || *p == 'e' || *p == 'E')) return false;
    v = neg ? -x : x;
    return true;
  }
  bool int32v(int32_t& v) {
    int64_t x;
    if (!integer(x) || x < INT32_MIN || x > INT32_MAX) return false;
    v = (int32_t)x;
    return true;
  }
  // a string of printable ASCII without escapes
  bool plain_string(std::string_view& s) {
    ws();
    if (p >= end || *p != '"') return false;
    const char* b = ++p;
    while (p < end && *p != '"') {
      const unsigned char c = (unsigned char)*p;
      if (c == '\\' || c < 0x20 || c >= 0x80) return false;
      ++p;
    }
    if (p >= end) return false;
    s = std::string_view(b, (size_t)(p - b));
    ++p;
    return true;
  }
  bool skip_string() {
    ws();
    if (p >= end || *p != '"') return false;
    ++p;
    while (p < end) {
      const unsigned char c = (unsigned char)*p;
      if (c == '"') {
        ++p;
        return true;
      }
      if (c < 0x20 || c >= 0x80) return false;
      if (c == '\\') {
        if (p + 1 >= end) return false;
        p += 2;
        continue;
      }
      ++p;
    }
    return false;
  }
  bool skip_number() {
    ws();
    const char* s = p;
    if (p < end && *p == '-') ++p;
    if (p >= end || *p < '0' || *p > '9') return false;
    if (*p == '0') {
      ++p;
    } else {
      while (p < end && *p >= '0' && *p <= '9') ++p;
    }
    if (p < end && *p == '.') {
      ++p;
      const char* d = p;
      while (p < end && *p >= '0' && *p <= '9') ++p;
      if (p == d) return false;
    }
    if (p < end && (*p == 'e' || *p == 'E')) {
      ++p;
      if (p < end && (*p == '+' || *p == '-')) ++p;
      const char* d = p;
      while (p < end && *p >= '0' && *p <= '9') ++p;
      if (p == d) return false;
    }
    return p > s;
  }
  bool skip_value(int depth) {
    if (depth > 64) return false;
    ws();
    if (p >= end) return false;
    const char c = *p;
    if (c == '"') return skip_string();
    if (c == '{') {
      ++p;
      if (lit('}')) return true;
      do {
        if (!skip_string() || !lit(':') || !skip_value(depth + 1)) return false;
      } while (lit(','));
      return lit('}');
    }
    if (c == '[') {
      ++p;
      if (lit(']')) return true;
      do {
        if (!skip_value(depth + 1)) return false;
      } while (lit(','));
      return lit(']');
    }
    if (keyword("true") || keyword("false") || keyword("null")) return true;
    return skip_number();
  }
  // null | [int, ...]
  bool int_list(std::vector<int32_t>& out) {
    if (keyword("null")) return true;
    if (!lit('[')) return false;
    if (lit(']')) return true;
    do {
      int32_t v;
      if (!int32v(v)) return false;
      out.push_back(v);
    } while (lit(','));
    return lit(']');
  }
  // null | [[int, null | [int, ...]], ...]
  bool adj_list(SkelOut& out) {
    if (keyword("null")) return true;
    if (!lit('[')) return false;
    if (lit(']')) return true;
    do {
      int32_t u;
      if (!lit('[') || !int32v(u) || !lit(',') || !int_list(out.eids) || !lit(']')) return false;
      out.adj_u.push_back(u);
      out.adj_eptr.push_back((int64_t)out.eids.size());
    } while (lit(','));
    return lit(']');
  }

  bool line(SkelOut& out) {
    if (!lit('{')) return false;
    bool has_src = false, has_topo = false, has_adj = false, has_dn = false, has_rc = false;
    int64_t src = 0;
    int8_t rc = -1;
    if (!lit('}')) {
      do {
        std::string_view key;
        if (!plain_string(key) || !lit(':')) return false;
        if (key == "src_net") return false;  // name form: the Python reader
        if (key == "src_net_idx") {
          if (has_src || !integer(src)) return false;
          has_src = true;
        } else if (key == "cone_topo_net_idxs") {
          if (has_topo || !int_list(out.topo)) return false;
          has_topo = true;
        } else if (key == "cone_adj_idx") {
          if (has_adj || !adj_list(out)) return false;
          has_adj = true;
        } else if (key == "reachable_dnet_idxs") {
          if (has_dn || !int_list(out.dn)) return false;
          has_dn = true;
        } else if (key == "has_reconv") {
          if (has_rc) return false;
          has_rc = true;
          if (keyword("true")) rc = 1;
          else if (keyword("false")) rc = 0;
          else if (keyword("null")) rc = -1;
          else return false;
        } else {
          if (!skip_value(0)) return false;
        }
      } while (lit(','));
      if (!lit('}')) return false;
    }
    ws();
    if (p != end || !has_src) return false;
    out.src.push_back(src);
    out.topo_ptr.push_back((int64_t)out.topo.size());
    out.adj_ptr.push_back((int64_t)out.adj_u.size());
    out.dn_ptr.push_back((int64_t)out.dn.size());
    out.reconv.push_back(rc);
    return true;
  }
};

// Parse the lines in [p, end) into out; false if any line needs the Python reader.
bool parse_lines(const char* p, const char* end, SkelOut& out) {
  while (p < end) {
    const char* nl = static_cast<const char*>(std::memchr(p, '\n', (size_t)(end - p)));
    const char* le = nl ? nl : end;
    const char* b = p;
    const char* e = le;
    while (b < e && (*b == ' ' || *b == '\t')) ++b;
    while (e > b && (e[-1] == ' ' || e[-1] == '\t')) --e;
    if (b < e) {
      LineParser lp{b, e};
      if (!lp.line(out)) return false;
    }
    p = nl ? nl + 1 : end;
  }
  return true;
}

template <typename T>
void append_shifted(std::vector<T>& dst, const std::vector<T>& src, T shift, size_t skip_first) {
  for (size_t i = skip_first; i < src.size(); i++) dst.push_back(src[i] + shift);
}

py::object parse_site_cones(const std::string& path, int n_threads) {
  std::string buf;
  {
    // a file that cannot be read goes to the Python reader too, which reports it
    std::ifstream f(path, std::ios::binary);
    if (!f) return py::none();
    f.seekg(0, std::ios::end);
    const std::streamoff n = f.tellg();
    if (n < 0) return py::none();
    f.seekg(0, std::ios::beg);
    buf.resize((size_t)n);
    if (n > 0 && !f.read(&buf[0], n)) return py::none();
  }
  SkelOut out;
  bool ok = true;
  {
    py::gil_scoped_release release;
    if (std::memchr(buf.data(), '\r', buf.size()) != nullptr) ok = false;  // Python splits lines differently
    // chunks that end at line ends, parsed in parallel and joined in order
    const int nt = std::max(1, std::min(n_threads, (int)(buf.size() >> 20) + 1));
    std::vector<const char*> cut{buf.data()};
    for (int t = 1; t < nt && ok; t++) {
      const char* q = buf.data() + buf.size() * (size_t)t / (size_t)nt;
      if (q < cut.back()) q = cut.back();
      const char* nl = static_cast<const char*>(std::memchr(q, '\n', (size_t)(buf.data() + buf.size() - q)));
      cut.push_back(nl ? nl + 1 : buf.data() + buf.size());
    }
    cut.push_back(buf.data() + buf.size());
    const size_t n_chunks = cut.size() - 1;
    std::vector<SkelOut> parts(n_chunks);
    std::vector<char> part_ok(n_chunks, 1);
    if (ok) {
      std::vector<std::exception_ptr> errors(n_chunks);
      std::atomic<bool> stop{false};  // each worker parses one chunk: nothing to stop
      auto work = [&](int c) {
        try {
          part_ok[(size_t)c] = parse_lines(cut[(size_t)c], cut[(size_t)c + 1], parts[(size_t)c]);
        } catch (...) {
          errors[(size_t)c] = std::current_exception();
        }
      };
      run_workers((int)n_chunks, stop, work);
      for (auto& e : errors)
        if (e) std::rethrow_exception(e);
      for (char x : part_ok) ok = ok && x;
    }
    if (ok && n_chunks == 1) {
      out = std::move(parts[0]);
    } else if (ok) {
      for (auto& q : parts) {
        const int64_t t0 = (int64_t)out.topo.size(), a0 = (int64_t)out.adj_u.size();
        const int64_t e0 = (int64_t)out.eids.size(), d0 = (int64_t)out.dn.size();
        out.src.insert(out.src.end(), q.src.begin(), q.src.end());
        out.reconv.insert(out.reconv.end(), q.reconv.begin(), q.reconv.end());
        out.topo.insert(out.topo.end(), q.topo.begin(), q.topo.end());
        out.adj_u.insert(out.adj_u.end(), q.adj_u.begin(), q.adj_u.end());
        out.eids.insert(out.eids.end(), q.eids.begin(), q.eids.end());
        out.dn.insert(out.dn.end(), q.dn.begin(), q.dn.end());
        append_shifted(out.topo_ptr, q.topo_ptr, t0, 1);
        append_shifted(out.adj_ptr, q.adj_ptr, a0, 1);
        append_shifted(out.adj_eptr, q.adj_eptr, e0, 1);
        append_shifted(out.dn_ptr, q.dn_ptr, d0, 1);
        q = SkelOut();
      }
    }
  }
  buf.clear();
  buf.shrink_to_fit();
  if (!ok) return py::none();
  py::dict r;
  r["src"] = move_to_array(std::move(out.src));
  r["topo_ptr"] = move_to_array(std::move(out.topo_ptr));
  r["topo"] = move_to_array(std::move(out.topo));
  r["adj_ptr"] = move_to_array(std::move(out.adj_ptr));
  r["adj_u"] = move_to_array(std::move(out.adj_u));
  r["adj_eptr"] = move_to_array(std::move(out.adj_eptr));
  r["eids"] = move_to_array(std::move(out.eids));
  r["dn_ptr"] = move_to_array(std::move(out.dn_ptr));
  r["dn"] = move_to_array(std::move(out.dn));
  r["has_reconv"] = move_to_array(std::move(out.reconv));
  return r;
}

// ---------------------------------------------------------------------------
// ConeCompiler
// ---------------------------------------------------------------------------
template <typename T>
struct Span {
  const T* p = nullptr;
  int64_t n = 0;
  const T& operator[](int64_t i) const { return p[i]; }
};

// status of one compiled site
enum CompileStatus {
  CS_OK = 0,
  CS_UNKNOWN_EDGE = 1,
  CS_NO_CELL_FUNCTION = 2,
  CS_NO_TRUTH_TABLE = 3,
  CS_PIN_NOT_CONNECTED = 4,
  CS_DNET_NOT_IN_CONE = 5,
  CS_SITE_NOT_IN_CONE = 6,
  CS_TOO_LARGE = 7,
};

// Integer tables built by setfi.campaign.fastpath.compile_tables.  Names
// (nets), pins and instances are numbered there; an edge row is an entry of
// edges.json; a cell entry is a dict entry of cell_arcs.json.
struct Tables {
  int64_t n_names = 0, n_gkeys = 0, n_ffid = 0;
  Span<int32_t> rank, gidx;                                           // per name
  Span<int32_t> eid_row;                                              // eid -> edge row or -1
  Span<int32_t> e_src, e_dst, e_gkey, e_inst, e_inpin, e_mask, e_dkey, e_cause;  // per edge row
  Span<uint8_t> ca_valid;                                             // per cell entry
  Span<int64_t> ca_vars_ptr, ca_truth_ptr, ca_side_ptr;
  Span<int32_t> ca_vars, ca_side;
  Span<uint8_t> ca_truth;
  Span<int64_t> ip_ptr;                                               // per instance: pin -> name
  Span<int32_t> ip_pin, ip_nid;
  Span<uint8_t> inst_empty;
  Span<int64_t> wl_ptr;                                               // per instance: pin -> wire
  Span<int32_t> wl_pin;
  Span<int64_t> wl_rise, wl_fall;
  bool has_wires = false;
  Span<int64_t> df_ptr;                                               // per name: observed FFs on it
  Span<int32_t> df_ffid;
  Span<int64_t> sk_topo_ptr, sk_adj_ptr, sk_adj_eptr, sk_dn_ptr;      // skeleton rows
  Span<int32_t> sk_topo, sk_eids, sk_dn;
};

struct CompileWS {
  int32_t stamp = 0;
  std::vector<int32_t> topo_mark, all_mark, local_mark, local_of;  // per name
  std::vector<int32_t> gkey_mark, gkey_gid;                        // per gate key
  std::vector<int32_t> ff_mark, ff_li;                             // per ffid
  std::vector<int32_t> gate_row, raw_gid, raw_pin, raw_row, fo_src, fo_gid;
  std::vector<int32_t> cnt, grp_pin, grp_row, arc_ptr, arc_pin, arc_row;
  std::vector<int32_t> extra, net_nid;
  std::vector<int32_t> g_cell, g_inst, g_out, g_var_ptr;
  std::vector<int32_t> v_pin, v_net, v_src, v_dkey, v_cause, v_side_ptr, v_side;
  std::vector<int8_t> v_mode;
  std::vector<std::vector<int32_t>> fanout;
  std::vector<std::tuple<int32_t, int32_t, int64_t, int64_t>> wires;  // drv, sink, rise, fall
  std::vector<int32_t> ffs;

  void init(const Tables& T) {
    topo_mark.assign((size_t)T.n_names, 0);
    all_mark.assign((size_t)T.n_names, 0);
    local_mark.assign((size_t)T.n_names, 0);
    local_of.assign((size_t)T.n_names, 0);
    gkey_mark.assign((size_t)T.n_gkeys, 0);
    gkey_gid.assign((size_t)T.n_gkeys, 0);
    ff_mark.assign((size_t)T.n_ffid, 0);
    ff_li.assign((size_t)T.n_ffid, 0);
    stamp = 0;
  }
  int32_t next_stamp() {
    if (stamp >= INT32_MAX - 1) {
      std::fill(topo_mark.begin(), topo_mark.end(), 0);
      std::fill(all_mark.begin(), all_mark.end(), 0);
      std::fill(local_mark.begin(), local_mark.end(), 0);
      std::fill(gkey_mark.begin(), gkey_mark.end(), 0);
      std::fill(ff_mark.begin(), ff_mark.end(), 0);
      stamp = 0;
    }
    return ++stamp;
  }
};

class ConeCompiler {
 public:
  explicit ConeCompiler(py::dict tables) : keep_(tables) {
    T_.n_names = py::cast<int64_t>(tables["n_names"]);
    T_.n_gkeys = py::cast<int64_t>(tables["n_gkeys"]);
    T_.n_ffid = py::cast<int64_t>(tables["n_ffid"]);
    T_.has_wires = py::cast<bool>(tables["has_wires"]);
    bind(T_.rank, "rank");
    bind(T_.gidx, "gidx");
    bind(T_.eid_row, "eid_row");
    bind(T_.e_src, "e_src");
    bind(T_.e_dst, "e_dst");
    bind(T_.e_gkey, "e_gkey");
    bind(T_.e_inst, "e_inst");
    bind(T_.e_inpin, "e_inpin");
    bind(T_.e_mask, "e_mask");
    bind(T_.e_dkey, "e_dkey");
    bind(T_.e_cause, "e_cause");
    bind(T_.ca_valid, "ca_valid");
    bind(T_.ca_vars_ptr, "ca_vars_ptr");
    bind(T_.ca_vars, "ca_vars");
    bind(T_.ca_truth_ptr, "ca_truth_ptr");
    bind(T_.ca_truth, "ca_truth");
    bind(T_.ca_side_ptr, "ca_side_ptr");
    bind(T_.ca_side, "ca_side");
    bind(T_.ip_ptr, "ip_ptr");
    bind(T_.ip_pin, "ip_pin");
    bind(T_.ip_nid, "ip_nid");
    bind(T_.inst_empty, "inst_empty");
    bind(T_.wl_ptr, "wl_ptr");
    bind(T_.wl_pin, "wl_pin");
    bind(T_.wl_rise, "wl_rise");
    bind(T_.wl_fall, "wl_fall");
    bind(T_.df_ptr, "df_ptr");
    bind(T_.df_ffid, "df_ffid");
    bind(T_.sk_topo_ptr, "sk_topo_ptr");
    bind(T_.sk_topo, "sk_topo");
    bind(T_.sk_adj_ptr, "sk_adj_ptr");
    bind(T_.sk_adj_eptr, "sk_adj_eptr");
    bind(T_.sk_eids, "sk_eids");
    bind(T_.sk_dn_ptr, "sk_dn_ptr");
    bind(T_.sk_dn, "sk_dn");
    check();
  }

  // Compile the skeleton rows `rows` (site name ids `site_nids`).  Returns
  // (list of CompiledCone or None, int32 status per site).
  py::tuple compile(I64A rows, I32A site_nids, int n_threads, bool keep_names) {
    const int64_t n = (int64_t)rows.size();
    if ((int64_t)site_nids.size() != n) throw std::runtime_error("ConeCompiler.compile: rows / site_nids length");
    const int64_t* rp = rows.data();
    const int32_t* sp = site_nids.data();
    const int64_t n_rows = T_.sk_topo_ptr.n - 1;
    for (int64_t i = 0; i < n; i++) {
      if (rp[i] < 0 || rp[i] >= n_rows) throw std::out_of_range("ConeCompiler.compile: skeleton row");
      if (sp[i] < 0 || sp[i] >= T_.n_names) throw std::out_of_range("ConeCompiler.compile: site name id");
    }
    std::vector<std::unique_ptr<CompiledCone>> res((size_t)n);
    std::vector<int32_t> status((size_t)n, 0);
    int nt = std::max(1, n_threads);
    if ((int64_t)nt > n) nt = (int)std::max<int64_t>(1, n);
    std::atomic<int64_t> next{0};
    std::atomic<bool> stop{false};
    bool interrupted = false;  // a Python signal handler raised (set by thread 0, the caller)
    SignalPoll poll;
    std::vector<std::exception_ptr> errors((size_t)nt);
    auto work = [&](int tid) {
      try {
        CompileWS ws;
        ws.init(T_);
        for (;;) {
          if (stop.load()) return;
          if (tid == 0 && poll.pending()) {
            interrupted = true;
            stop.store(true);
            return;
          }
          const int64_t i = next.fetch_add(1);
          if (i >= n) return;
          auto c = std::make_unique<CompiledCone>();
          const int st = compile_one(rp[i], sp[i], ws, *c, keep_names);
          status[(size_t)i] = st;
          if (st == CS_OK) res[(size_t)i] = std::move(c);
        }
      } catch (...) {
        errors[(size_t)tid] = std::current_exception();
      }
    };
    {
      py::gil_scoped_release release;
      run_workers(nt, stop, work);
    }
    if (interrupted) throw py::error_already_set();
    for (auto& e : errors)
      if (e) std::rethrow_exception(e);
    py::list cones;
    for (int64_t i = 0; i < n; i++) {
      if (res[(size_t)i]) cones.append(py::cast(res[(size_t)i].release(), py::return_value_policy::take_ownership));
      else cones.append(py::none());
    }
    return py::make_tuple(cones, to_array(status));
  }

 private:
  py::dict keep_;
  Tables T_;

  template <typename T>
  void bind(Span<T>& s, const char* key) {
    if (!keep_.contains(key)) throw std::runtime_error(std::string("ConeCompiler: missing table '") + key + "'");
    auto a = py::cast<py::array_t<T, py::array::c_style | py::array::forcecast>>(keep_[key]);
    if (a.ndim() != 1) throw std::runtime_error(std::string("ConeCompiler: table '") + key + "' must be 1-D");
    keep_[key] = a;  // keep the (possibly converted) array alive
    s.p = a.data();
    s.n = (int64_t)a.size();
  }

  void check() const {
    auto need = [](bool ok, const char* what) {
      if (!ok) throw std::runtime_error(std::string("ConeCompiler: inconsistent tables (") + what + ")");
    };
    need(T_.rank.n == T_.n_names && T_.gidx.n == T_.n_names, "names");
    need(T_.df_ptr.n == T_.n_names + 1, "df_ptr");
    const int64_t ne = T_.e_src.n;
    for (auto* s : {&T_.e_dst, &T_.e_gkey, &T_.e_inst, &T_.e_inpin, &T_.e_mask, &T_.e_dkey, &T_.e_cause})
      need(s->n == ne, "edge columns");
    for (int64_t i = 0; i < T_.eid_row.n; i++) need(T_.eid_row[i] < ne, "eid_row");
    const int64_t n_inst = T_.inst_empty.n;
    const int64_t n_cell = T_.ca_valid.n;
    for (int64_t r = 0; r < ne; r++) {
      need(T_.e_src[r] >= 0 && T_.e_src[r] < T_.n_names && T_.e_dst[r] >= 0 && T_.e_dst[r] < T_.n_names,
           "edge nets");
      need(T_.e_cause[r] >= -1 && T_.e_cause[r] < T_.n_names, "edge cause");
      need(T_.e_gkey[r] >= 0 && T_.e_gkey[r] < T_.n_gkeys, "edge gate key");
      need(T_.e_inst[r] >= 0 && T_.e_inst[r] < n_inst, "edge instance");
      need(T_.e_mask[r] >= -1 && T_.e_mask[r] < n_cell, "edge cell entry");
    }
    need(T_.ca_vars_ptr.n == n_cell + 1 && T_.ca_truth_ptr.n == n_cell + 1 && T_.ca_side_ptr.n == n_cell + 1,
         "cell entries");
    need(T_.ip_ptr.n == n_inst + 1, "ip_ptr");
    need(!T_.has_wires || T_.wl_ptr.n == n_inst + 1, "wl_ptr");
    for (int64_t k = 0; k < T_.ip_nid.n; k++) need(T_.ip_nid[k] >= -1 && T_.ip_nid[k] < T_.n_names, "ip_nid");
    for (int64_t k = 0; k < T_.df_ffid.n; k++) need(T_.df_ffid[k] >= 0 && T_.df_ffid[k] < T_.n_ffid, "df_ffid");
    const int64_t n_rows = T_.sk_topo_ptr.n - 1;
    need(n_rows >= 0 && T_.sk_adj_ptr.n == n_rows + 1 && T_.sk_dn_ptr.n == n_rows + 1, "skeleton rows");
    for (int64_t i = 0; i < T_.sk_topo.n; i++) need(T_.sk_topo[i] >= 0 && T_.sk_topo[i] < T_.n_names, "sk_topo");
    for (int64_t i = 0; i < T_.sk_dn.n; i++) need(T_.sk_dn[i] >= 0 && T_.sk_dn[i] < T_.n_names, "sk_dn");
  }

  int32_t pin_net(int32_t inst, int32_t pin) const {
    for (int64_t k = T_.ip_ptr[inst]; k < T_.ip_ptr[inst + 1]; k++)
      if (T_.ip_pin[k] == pin) return T_.ip_nid[k];
    return -1;
  }

  bool wire_of(int32_t inst, int32_t pin, int64_t& rise, int64_t& fall) const {
    for (int64_t k = T_.wl_ptr[inst]; k < T_.wl_ptr[inst + 1]; k++)
      if (T_.wl_pin[k] == pin) {
        rise = T_.wl_rise[k];
        fall = T_.wl_fall[k];
        return true;
      }
    return false;
  }

  // build_cone + pack_cone + build_capture_plan + binding data for one site.
  int compile_one(int64_t row, int32_t site_nid, CompileWS& W, CompiledCone& C, bool keep_names) const {
    const Tables& T = T_;
    const int32_t st = W.next_stamp();
    const int64_t ta = T.sk_topo_ptr[row], tb = T.sk_topo_ptr[row + 1];
    for (int64_t i = ta; i < tb; i++) W.topo_mark[(size_t)T.sk_topo[i]] = st;

    // gates in order of first appearance; their arcs and the fan-out pairs
    W.gate_row.clear();
    W.raw_gid.clear();
    W.raw_pin.clear();
    W.raw_row.clear();
    W.fo_src.clear();
    W.fo_gid.clear();
    int32_t n_gates = 0;
    const int64_t ea = T.sk_adj_eptr[T.sk_adj_ptr[row]], eb = T.sk_adj_eptr[T.sk_adj_ptr[row + 1]];
    for (int64_t i = ea; i < eb; i++) {
      const int64_t eid = T.sk_eids[i];
      if (eid < 0 || eid >= T.eid_row.n) return CS_UNKNOWN_EDGE;
      const int32_t r = T.eid_row[eid];
      if (r < 0) return CS_UNKNOWN_EDGE;
      if (W.topo_mark[(size_t)T.e_dst[r]] != st) continue;
      const int32_t gk = T.e_gkey[r];
      int32_t gid;
      if (W.gkey_mark[(size_t)gk] != st) {
        W.gkey_mark[(size_t)gk] = st;
        gid = n_gates++;
        W.gkey_gid[(size_t)gk] = gid;
        W.gate_row.push_back(r);
      } else {
        gid = W.gkey_gid[(size_t)gk];
      }
      W.raw_gid.push_back(gid);
      W.raw_pin.push_back(T.e_inpin[r]);
      W.raw_row.push_back(r);
      W.fo_src.push_back(T.e_src[r]);
      W.fo_gid.push_back(gid);
    }

    // per gate, the arcs keyed by input pin: first-seen order, last edge wins
    const size_t n_raw = W.raw_gid.size();
    W.cnt.assign((size_t)n_gates + 1, 0);
    for (int32_t g : W.raw_gid) W.cnt[(size_t)g + 1]++;
    for (int32_t g = 0; g < n_gates; g++) W.cnt[(size_t)g + 1] += W.cnt[(size_t)g];
    W.grp_pin.resize(n_raw);
    W.grp_row.resize(n_raw);
    {
      std::vector<int32_t>& pos = W.extra;  // reused as scratch
      pos.assign(W.cnt.begin(), W.cnt.end() - 1);
      for (size_t k = 0; k < n_raw; k++) {
        const int32_t at = pos[(size_t)W.raw_gid[k]]++;
        W.grp_pin[(size_t)at] = W.raw_pin[k];
        W.grp_row[(size_t)at] = W.raw_row[k];
      }
    }
    W.arc_ptr.assign((size_t)n_gates + 1, 0);
    W.arc_pin.clear();
    W.arc_row.clear();
    for (int32_t g = 0; g < n_gates; g++) {
      const size_t base = W.arc_pin.size();
      for (int32_t k = W.cnt[(size_t)g]; k < W.cnt[(size_t)g + 1]; k++) {
        const int32_t pin = W.grp_pin[(size_t)k];
        size_t q = base;
        while (q < W.arc_pin.size() && W.arc_pin[q] != pin) q++;
        if (q < W.arc_pin.size()) {
          W.arc_row[q] = W.grp_row[(size_t)k];
        } else {
          W.arc_pin.push_back(pin);
          W.arc_row.push_back(W.grp_row[(size_t)k]);
        }
      }
      W.arc_ptr[(size_t)g + 1] = (int32_t)W.arc_pin.size();
    }

    // per gate: cell function, input nets, and per input its arc data;
    // collect the nets that are not in the topological list
    W.extra.clear();
    auto mark_net = [&](int32_t nid) {
      if (W.topo_mark[(size_t)nid] != st && W.all_mark[(size_t)nid] != st) {
        W.all_mark[(size_t)nid] = st;
        W.extra.push_back(nid);
      }
    };
    W.g_cell.resize((size_t)n_gates);
    W.g_inst.resize((size_t)n_gates);
    W.g_out.resize((size_t)n_gates);
    W.g_var_ptr.assign((size_t)n_gates + 1, 0);
    W.v_pin.clear();
    W.v_net.clear();
    W.v_dkey.clear();
    W.v_cause.clear();
    W.v_mode.clear();
    W.v_side_ptr.assign(1, 0);
    W.v_side.clear();
    int64_t unconnected = 0;
    for (int32_t g = 0; g < n_gates; g++) {
      int32_t cell = -1;
      for (int32_t a = W.arc_ptr[(size_t)g]; a < W.arc_ptr[(size_t)g + 1]; a++) {
        const int32_t m = T.e_mask[W.arc_row[(size_t)a]];
        if (m >= 0) {
          cell = m;
          break;
        }
      }
      if (cell < 0) return CS_NO_CELL_FUNCTION;
      if (!T.ca_valid[cell]) return CS_NO_TRUTH_TABLE;
      const int32_t r0 = W.gate_row[(size_t)g];
      const int32_t inst = T.e_inst[r0];
      W.g_cell[(size_t)g] = cell;
      W.g_inst[(size_t)g] = inst;
      W.g_out[(size_t)g] = T.e_dst[r0];
      const int64_t va = T.ca_vars_ptr[cell], vb = T.ca_vars_ptr[cell + 1];
      const size_t v0 = W.v_net.size();
      for (int64_t q = va; q < vb; q++) {
        const int32_t nid = pin_net(inst, T.ca_vars[q]);
        if (nid < 0) return CS_PIN_NOT_CONNECTED;
        W.v_pin.push_back(T.ca_vars[q]);
        W.v_net.push_back(nid);
      }
      for (int64_t q = va; q < vb; q++) {
        const int32_t pin = T.ca_vars[q];
        int32_t arow = -1;
        for (int32_t a = W.arc_ptr[(size_t)g]; a < W.arc_ptr[(size_t)g + 1]; a++)
          if (W.arc_pin[(size_t)a] == pin) {
            arow = W.arc_row[(size_t)a];
            break;
          }
        if (arow < 0) {
          W.v_dkey.push_back(-1);
          W.v_cause.push_back(-1);
          W.v_mode.push_back(0);
          W.v_side_ptr.push_back((int32_t)W.v_side.size());
          continue;
        }
        W.v_dkey.push_back(T.e_dkey[arow]);
        W.v_cause.push_back(T.e_cause[arow]);
        const int32_t m = T.e_mask[arow];
        const int64_t sa = (m >= 0) ? T.ca_side_ptr[m] : 0, sb = (m >= 0) ? T.ca_side_ptr[m + 1] : 0;
        if (sa == sb) {
          W.v_mode.push_back(1);
        } else {
          const size_t keep = W.v_side.size();
          bool all = true;
          for (int64_t s = sa; s < sb; s++) {
            const int32_t nn = pin_net(inst, T.ca_side[s]);
            if (nn < 0) {
              unconnected++;
              all = false;
              break;
            }
            W.v_side.push_back(nn);
          }
          if (all) {
            W.v_mode.push_back(2);
          } else {
            W.v_side.resize(keep);
            W.v_mode.push_back(0);
          }
        }
        W.v_side_ptr.push_back((int32_t)W.v_side.size());
      }
      W.g_var_ptr[(size_t)g + 1] = (int32_t)W.v_net.size();
      mark_net(W.g_out[(size_t)g]);
      for (size_t q = v0; q < W.v_net.size(); q++) mark_net(W.v_net[q]);
      for (size_t q = v0; q < W.v_net.size(); q++)
        if (W.v_cause[q] >= 0) mark_net(W.v_cause[q]);
      for (size_t q = v0; q < W.v_net.size(); q++)
        if (W.v_mode[q] == 2)
          for (int32_t s = W.v_side_ptr[q]; s < W.v_side_ptr[q + 1]; s++) mark_net(W.v_side[(size_t)s]);
    }

    // local numbering: the topological list, then the other nets by name
    W.net_nid.clear();
    for (int64_t i = ta; i < tb; i++) {
      const int32_t nid = T.sk_topo[i];
      if (W.local_mark[(size_t)nid] != st) {
        W.local_mark[(size_t)nid] = st;
        W.local_of[(size_t)nid] = (int32_t)W.net_nid.size();
        W.net_nid.push_back(nid);
      }
    }
    std::sort(W.extra.begin(), W.extra.end(),
              [&](int32_t a, int32_t b) { return T.rank[a] < T.rank[b]; });
    for (int32_t nid : W.extra) {
      W.local_mark[(size_t)nid] = st;
      W.local_of[(size_t)nid] = (int32_t)W.net_nid.size();
      W.net_nid.push_back(nid);
    }
    if (W.net_nid.size() >= (size_t)INT32_MAX / 2) return CS_TOO_LARGE;
    auto local = [&](int32_t nid) -> int32_t { return W.local_of[(size_t)nid]; };

    // fan-out lists, in edge order
    size_t n_local = W.net_nid.size();
    if (W.fanout.size() < n_local) W.fanout.resize(n_local);
    for (size_t i = 0; i < n_local; i++) W.fanout[i].clear();
    for (size_t k = 0; k < W.fo_src.size(); k++) {
      const int32_t s = W.fo_src[k];
      if (W.local_mark[(size_t)s] == st) W.fanout[(size_t)local(s)].push_back(W.fo_gid[k]);
    }

    // per input: local nets (v_net, spliced by wires below), driver nets
    // before splicing (v_src), causes and side nets
    const size_t n_vars = W.v_net.size();
    for (size_t q = 0; q < n_vars; q++) W.v_net[q] = local(W.v_net[q]);
    W.v_src.assign(W.v_net.begin(), W.v_net.end());
    for (size_t q = 0; q < n_vars; q++)
      if (W.v_cause[q] >= 0) W.v_cause[q] = local(W.v_cause[q]);
    for (size_t s = 0; s < W.v_side.size(); s++) W.v_side[s] = local(W.v_side[s]);

    // insert_transport_nodes
    W.wires.clear();
    if (T.has_wires) {
      std::map<std::tuple<int32_t, int32_t, int32_t>, int32_t> pin_node;  // (drv, inst, pin) -> sink
      for (int32_t g = 0; g < n_gates; g++) {
        const int32_t inst = W.g_inst[(size_t)g];
        if (T.inst_empty[inst]) continue;
        const int32_t qa = W.g_var_ptr[(size_t)g], qb = W.g_var_ptr[(size_t)g + 1];
        for (int32_t q = qa; q < qb; q++) {
          int64_t rise = 0, fall = 0;
          if (!wire_of(inst, W.v_pin[(size_t)q], rise, fall)) continue;
          const int32_t drv = W.v_net[(size_t)q];
          if (drv < 0) continue;
          const auto key = std::make_tuple(drv, inst, W.v_pin[(size_t)q]);
          auto it = pin_node.find(key);
          int32_t sink;
          if (it == pin_node.end()) {
            sink = (int32_t)n_local++;
            pin_node.emplace(key, sink);
            W.net_nid.push_back(-1);
            if (W.fanout.size() < n_local) W.fanout.resize(n_local);
            W.fanout[(size_t)sink].clear();
            W.wires.emplace_back(drv, sink, rise, fall);
          } else {
            sink = it->second;
          }
          W.v_net[(size_t)q] = sink;
          if (W.v_cause[(size_t)q] == drv) W.v_cause[(size_t)q] = sink;
          for (int32_t q2 = qa; q2 < qb; q2++) {
            if (W.v_mode[(size_t)q2] != 2) continue;
            for (int32_t s = W.v_side_ptr[(size_t)q2]; s < W.v_side_ptr[(size_t)q2 + 1]; s++)
              if (W.v_side[(size_t)s] == drv) W.v_side[(size_t)s] = sink;
          }
          auto& fs = W.fanout[(size_t)sink];
          if (std::find(fs.begin(), fs.end(), g) == fs.end()) fs.push_back(g);
        }
      }
      for (int32_t g = 0; g < n_gates; g++) {
        const int32_t qa = W.g_var_ptr[(size_t)g], qb = W.g_var_ptr[(size_t)g + 1];
        for (int32_t q = qa; q < qb; q++) {
          const int32_t drv = W.v_src[(size_t)q];
          if (drv < 0 || (size_t)drv >= n_local) continue;
          bool dup = false;  // each driver once (a set in the Python code)
          for (int32_t q2 = qa; q2 < q && !dup; q2++) dup = (W.v_src[(size_t)q2] == drv);
          if (dup) continue;
          bool still = false;
          for (int32_t q2 = qa; q2 < qb && !still; q2++) still = (W.v_net[(size_t)q2] == drv);
          if (still) continue;
          auto& fd = W.fanout[(size_t)drv];
          fd.erase(std::remove(fd.begin(), fd.end(), g), fd.end());
        }
      }
    }

    // record slots of the reachable D nets
    const int64_t da = T.sk_dn_ptr[row], db = T.sk_dn_ptr[row + 1];
    C.rec_idx.assign(n_local, -1);
    int32_t n_rec = 0;
    for (int64_t i = da; i < db; i++) {
      const int32_t nid = T.sk_dn[i];
      if (W.local_mark[(size_t)nid] != st) return CS_DNET_NOT_IN_CONE;
      const int32_t ni = local(nid);
      if (C.rec_idx[(size_t)ni] < 0) C.rec_idx[(size_t)ni] = n_rec++;
    }
    if (W.local_mark[(size_t)site_nid] != st) return CS_SITE_NOT_IN_CONE;
    C.src_i = local(site_nid);

    // pack_cone
    {
      size_t nnz = 0;
      for (size_t i = 0; i < n_local; i++) nnz += W.fanout[i].size();
      C.fanout_ptr.resize(n_local + 1);
      C.fanout_idx.resize(nnz);
      size_t at = 0;
      for (size_t i = 0; i < n_local; i++) {
        C.fanout_ptr[i] = (int32_t)at;
        for (int32_t g : W.fanout[i]) C.fanout_idx[at++] = g;
      }
      C.fanout_ptr[n_local] = (int32_t)at;
    }
    C.gate_out.resize((size_t)n_gates);
    C.gate_var_ptr.assign(W.g_var_ptr.begin(), W.g_var_ptr.begin() + n_gates + 1);
    C.gate_truth_ptr.resize((size_t)n_gates + 1);
    C.gate_truth_ptr[0] = 0;
    size_t n_truth = 0;
    for (int32_t g = 0; g < n_gates; g++) {
      const int32_t cell = W.g_cell[(size_t)g];
      n_truth += (size_t)(T.ca_truth_ptr[cell + 1] - T.ca_truth_ptr[cell]);
      C.gate_truth_ptr[(size_t)g + 1] = (int32_t)n_truth;
      C.gate_out[(size_t)g] = local(W.g_out[(size_t)g]);
    }
    C.gate_truth.resize(n_truth);
    for (int32_t g = 0; g < n_gates; g++) {
      const int32_t cell = W.g_cell[(size_t)g];
      std::copy(T.ca_truth.p + T.ca_truth_ptr[cell], T.ca_truth.p + T.ca_truth_ptr[cell + 1],
                C.gate_truth.begin() + C.gate_truth_ptr[(size_t)g]);
    }
    C.gate_var_nets.assign(W.v_net.begin(), W.v_net.end());
    C.arc_dkey_id.assign(W.v_dkey.begin(), W.v_dkey.end());
    C.arc_cause_i.assign(W.v_cause.begin(), W.v_cause.end());
    C.arc_side_mode.assign(W.v_mode.begin(), W.v_mode.end());
    C.arc_side_ptr.assign(W.v_side_ptr.begin(), W.v_side_ptr.end());
    C.arc_side_nets.assign(W.v_side.begin(), W.v_side.end());
    if (!W.wires.empty()) {
      // _pack_wires: per driver, zero-delay wires first, each group in order
      C.wire_ptr.assign(n_local + 1, 0);
      C.wire_zsplit.assign(n_local, 0);
      std::vector<std::vector<size_t>> by_drv(n_local);
      for (size_t w = 0; w < W.wires.size(); w++) by_drv[(size_t)std::get<0>(W.wires[w])].push_back(w);
      for (size_t ni = 0; ni < n_local; ni++) {
        C.wire_ptr[ni] = (int32_t)C.wire_sink.size();
        for (int pass = 0; pass < 2; pass++) {
          for (size_t w : by_drv[ni]) {
            const bool zero = std::get<2>(W.wires[w]) == 0 && std::get<3>(W.wires[w]) == 0;
            if (zero != (pass == 0)) continue;
            C.wire_sink.push_back(std::get<1>(W.wires[w]));
            C.wire_rise_ps.push_back(std::get<2>(W.wires[w]));
            C.wire_fall_ps.push_back(std::get<3>(W.wires[w]));
          }
          if (pass == 0) C.wire_zsplit[ni] = (int32_t)C.wire_sink.size();
        }
      }
      C.wire_ptr[n_local] = (int32_t)C.wire_sink.size();
      for (auto& w : W.wires) {
        C.wp_drv.push_back(std::get<0>(w));
        C.wp_sink.push_back(std::get<1>(w));
      }
    }

    // build_capture_plan
    int64_t no_ff = 0;
    W.ffs.clear();
    for (int64_t i = da; i < db; i++) {
      const int32_t nid = T.sk_dn[i];
      const int64_t fa = T.df_ptr[nid], fb = T.df_ptr[nid + 1];
      if (fa == fb) {
        no_ff++;
        continue;
      }
      for (int64_t k = fa; k < fb; k++) {
        const int32_t f = T.df_ffid[k];
        if (W.ff_mark[(size_t)f] != st) {
          W.ff_mark[(size_t)f] = st;
          W.ffs.push_back(f);
        }
      }
    }
    std::sort(W.ffs.begin(), W.ffs.end());
    const int32_t n_ff = (int32_t)W.ffs.size();
    C.ffids.assign(W.ffs.begin(), W.ffs.end());
    for (int32_t li = 0; li < n_ff; li++) W.ff_li[(size_t)W.ffs[(size_t)li]] = li;
    C.dp_ptr.assign(1, 0);
    if (n_ff) {
      for (int64_t i = da; i < db; i++) {
        const int32_t nid = T.sk_dn[i];
        const int64_t fa = T.df_ptr[nid], fb = T.df_ptr[nid + 1];
        if (fa == fb) continue;
        const int32_t slot = C.rec_idx[(size_t)local(nid)];
        if (slot < 0) continue;
        C.dp_nid.push_back(nid);
        C.dp_rec_slot.push_back(slot);
        for (int64_t k = fa; k < fb; k++) C.dp_idx.push_back(W.ff_li[(size_t)T.df_ffid[k]]);
        C.dp_ptr.push_back((int32_t)C.dp_idx.size());
      }
    }

    // binding data
    C.gidx.resize(n_local);
    for (size_t i = 0; i < n_local; i++) {
      const int32_t nid = W.net_nid[i];
      C.gidx[i] = (nid >= 0) ? T.gidx[nid] : -1;
    }
    C.settle_in_ptr.assign(1, 0);
    for (int32_t g = 0; g < n_gates; g++) {
      if (C.gidx[(size_t)C.gate_out[(size_t)g]] >= 0) continue;
      C.settle_gate.push_back(g);
      for (int32_t q = W.g_var_ptr[(size_t)g]; q < W.g_var_ptr[(size_t)g + 1]; q++)
        C.settle_in.push_back(W.v_src[(size_t)q]);
      C.settle_in_ptr.push_back((int32_t)C.settle_in.size());
    }
    C.dp_g.resize(C.dp_nid.size());
    for (size_t k = 0; k < C.dp_nid.size(); k++) C.dp_g[k] = T.gidx[C.dp_nid[k]];
    if (keep_names) C.net_nid.assign(W.net_nid.begin(), W.net_nid.end());
    C.unconnected_side_pins = unconnected;
    C.dnets_without_observed_ff = no_ff;
    return CS_OK;
  }
};

// ---------------------------------------------------------------------------
// CompiledCone <-> numpy
// ---------------------------------------------------------------------------
py::dict cone_arrays(const CompiledCone& c) {
  py::dict d;
  const size_t n = c.rec_idx.size();
  d["fanout_ptr"] = to_array(c.fanout_ptr);
  d["fanout_idx"] = to_array(c.fanout_idx);
  d["gate_out"] = to_array(c.gate_out);
  d["gate_var_ptr"] = to_array(c.gate_var_ptr);
  d["gate_var_nets"] = to_array(c.gate_var_nets);
  d["gate_truth_ptr"] = to_array(c.gate_truth_ptr);
  d["gate_truth"] = to_array(c.gate_truth);
  d["arc_dkey_id"] = to_array(c.arc_dkey_id);
  d["arc_cause_i"] = to_array(c.arc_cause_i);
  d["arc_side_mode"] = to_array(c.arc_side_mode);
  d["arc_side_ptr"] = to_array(c.arc_side_ptr);
  d["arc_side_nets"] = to_array(c.arc_side_nets);
  d["rec_idx"] = to_array(c.rec_idx);
  if (c.wire_sink.empty()) {
    d["wire_ptr"] = to_array(std::vector<int32_t>(n + 1, 0));
    d["wire_zsplit"] = to_array(std::vector<int32_t>(n, 0));
  } else {
    d["wire_ptr"] = to_array(c.wire_ptr);
    d["wire_zsplit"] = to_array(c.wire_zsplit);
  }
  d["wire_sink"] = to_array(c.wire_sink);
  d["wire_rise_ps"] = to_array(c.wire_rise_ps);
  d["wire_fall_ps"] = to_array(c.wire_fall_ps);
  d["ffids"] = to_array(c.ffids);
  d["dp_rec_slot"] = to_array(c.dp_rec_slot);
  d["dp_ptr"] = to_array(c.dp_ptr);
  d["dp_idx"] = to_array(c.dp_idx);
  d["src_i"] = c.src_i;
  d["gidx"] = to_array(c.gidx);
  d["settle_gate"] = to_array(c.settle_gate);
  d["settle_in_ptr"] = to_array(c.settle_in_ptr);
  d["settle_in"] = to_array(c.settle_in);
  d["wp_drv"] = to_array(c.wp_drv);
  d["wp_sink"] = to_array(c.wp_sink);
  d["dp_g"] = to_array(c.dp_g);
  d["dp_nid"] = to_array(c.dp_nid);
  d["net_nid"] = to_array(c.net_nid);
  d["unconnected_side_pins"] = c.unconnected_side_pins;
  d["dnets_without_observed_ff"] = c.dnets_without_observed_ff;
  return d;
}

std::unique_ptr<CompiledCone> cone_from_arrays(const py::dict& d) {
  auto c = std::make_unique<CompiledCone>();
  c->fanout_ptr = from_array<int32_t>(d, "fanout_ptr");
  c->fanout_idx = from_array<int32_t>(d, "fanout_idx");
  c->gate_out = from_array<int32_t>(d, "gate_out");
  c->gate_var_ptr = from_array<int32_t>(d, "gate_var_ptr");
  c->gate_var_nets = from_array<int32_t>(d, "gate_var_nets");
  c->gate_truth_ptr = from_array<int32_t>(d, "gate_truth_ptr");
  c->gate_truth = from_array<uint8_t>(d, "gate_truth");
  c->arc_dkey_id = from_array<int32_t>(d, "arc_dkey_id");
  c->arc_cause_i = from_array<int32_t>(d, "arc_cause_i");
  c->arc_side_mode = from_array<int8_t>(d, "arc_side_mode");
  c->arc_side_ptr = from_array<int32_t>(d, "arc_side_ptr");
  c->arc_side_nets = from_array<int32_t>(d, "arc_side_nets");
  c->rec_idx = from_array<int32_t>(d, "rec_idx");
  c->wire_sink = from_array<int32_t>(d, "wire_sink");
  if (!c->wire_sink.empty()) {
    c->wire_ptr = from_array<int32_t>(d, "wire_ptr");
    c->wire_zsplit = from_array<int32_t>(d, "wire_zsplit");
    c->wire_rise_ps = from_array<int64_t>(d, "wire_rise_ps");
    c->wire_fall_ps = from_array<int64_t>(d, "wire_fall_ps");
  }
  c->ffids = from_array<int32_t>(d, "ffids");
  c->dp_rec_slot = from_array<int32_t>(d, "dp_rec_slot");
  c->dp_ptr = from_array<int32_t>(d, "dp_ptr");
  c->dp_idx = from_array<int32_t>(d, "dp_idx");
  c->src_i = py::cast<int32_t>(d["src_i"]);
  c->gidx = from_array<int32_t>(d, "gidx");
  c->settle_gate = from_array<int32_t>(d, "settle_gate");
  c->settle_in_ptr = from_array<int32_t>(d, "settle_in_ptr");
  c->settle_in = from_array<int32_t>(d, "settle_in");
  c->wp_drv = from_array<int32_t>(d, "wp_drv");
  c->wp_sink = from_array<int32_t>(d, "wp_sink");
  c->dp_g = from_array<int64_t>(d, "dp_g");
  if (d.contains("dp_nid")) c->dp_nid = from_array<int32_t>(d, "dp_nid");
  if (d.contains("net_nid")) c->net_nid = from_array<int32_t>(d, "net_nid");
  if (d.contains("unconnected_side_pins")) c->unconnected_side_pins = py::cast<int64_t>(d["unconnected_side_pins"]);
  if (d.contains("dnets_without_observed_ff"))
    c->dnets_without_observed_ff = py::cast<int64_t>(d["dnets_without_observed_ff"]);

  // the consistency the trial loop relies on
  const size_t n = c->rec_idx.size(), ng = c->gate_out.size(), nv = c->gate_var_nets.size();
  const size_t nff = c->ffids.size(), ndp = c->dp_rec_slot.size();
  auto need = [](bool ok, const char* what) {
    if (!ok) throw std::runtime_error(std::string("CompiledCone.from_arrays: inconsistent ") + what);
  };
  need(c->fanout_ptr.size() == n + 1 && c->gidx.size() == n, "net arrays");
  need(c->gate_var_ptr.size() == ng + 1 && c->gate_truth_ptr.size() == ng + 1, "gate arrays");
  need(c->arc_dkey_id.size() == nv && c->arc_cause_i.size() == nv && c->arc_side_mode.size() == nv &&
           c->arc_side_ptr.size() == nv + 1, "input arrays");
  need(c->wire_sink.empty() || (c->wire_ptr.size() == n + 1 && c->wire_zsplit.size() == n &&
                                c->wire_rise_ps.size() == c->wire_sink.size() &&
                                c->wire_fall_ps.size() == c->wire_sink.size()), "wire arrays");
  need(c->dp_ptr.size() == ndp + 1 && c->dp_g.size() == ndp, "D-net arrays");
  need(c->settle_in_ptr.size() == c->settle_gate.size() + 1 && c->wp_drv.size() == c->wp_sink.size(),
       "binding arrays");
  need(c->src_i >= 0 && (size_t)c->src_i < n, "src_i");
  for (int32_t g : c->settle_gate) need(g >= 0 && (size_t)g < ng, "settle_gate");
  for (int32_t x : c->settle_in) need(x >= 0 && (size_t)x < n, "settle_in");
  for (size_t w = 0; w < c->wp_drv.size(); w++)
    need(c->wp_drv[w] >= 0 && (size_t)c->wp_drv[w] < n && c->wp_sink[w] >= 0 && (size_t)c->wp_sink[w] < n,
         "wire pairs");
  for (int32_t x : c->gate_out) need(x >= 0 && (size_t)x < n, "gate_out");
  for (size_t k = 0; k < ndp; k++)
    for (int32_t q = c->dp_ptr[k]; q < c->dp_ptr[k + 1]; q++)
      need(q >= 0 && (size_t)q < c->dp_idx.size() && c->dp_idx[(size_t)q] >= 0 &&
               (size_t)c->dp_idx[(size_t)q] < nff, "dp_idx");
  return c;
}

}  // namespace

void register_cone_compiler(py::module_& m) {
  py::class_<CompiledCone>(m, "CompiledCone",
                           "One injection site compiled for run_block (see setfi.campaign.fastpath).")
      .def_static("from_arrays", &cone_from_arrays, py::arg("arrays"),
                  "Build from the arrays of the Python reference (keys as in arrays()).")
      .def("arrays", &cone_arrays, "All arrays, with the dtypes of PackedCone / CapturePlan.")
      .def_property_readonly("n_nets", &CompiledCone::n_nets)
      .def_property_readonly("n_gates", &CompiledCone::n_gates)
      .def_property_readonly("n_ff", &CompiledCone::n_ff)
      .def_readonly("src_i", &CompiledCone::src_i)
      .def_readonly("unconnected_side_pins", &CompiledCone::unconnected_side_pins)
      .def_readonly("dnets_without_observed_ff", &CompiledCone::dnets_without_observed_ff);

  py::class_<ConeCompiler>(m, "ConeCompiler",
                           "Compiles injection sites from the integer tables of setfi.campaign.fastpath.")
      .def(py::init<py::dict>(), py::arg("tables"))
      .def("compile", &ConeCompiler::compile, py::arg("rows"), py::arg("site_nids"), py::arg("n_threads") = 1,
           py::arg("keep_names") = false,
           "(list of CompiledCone or None, int32 status): status 0 = compiled; otherwise the Python "
           "reference must handle the site.");

  m.def("parse_site_cones", &parse_site_cones, py::arg("path"), py::arg("n_threads") = 1,
        "Read site_cones.jsonl (index form) into CSR arrays; None if the file needs the Python reader.");
}

}  // namespace setfi_fast
