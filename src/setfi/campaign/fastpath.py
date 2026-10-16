"""Cone compilation in C++ for the trial loop (``setfi._engine.run_block``).

``compile_sites`` turns injection sites into ``_engine.CompiledCone`` objects:
the packed cone, the capture plan and the data that binds a recorded state to
the cone.  The C++ ``ConeCompiler`` re-implements ``cone.build_cone`` (with
``insert_transport_nodes``), ``cone.pack_cone``, ``cone.build_capture_plan`` and
the binding step of the trial loop on integer tables built here, and gives the
same arrays.  A site it does not handle, or that the Python code would reject,
is compiled by the Python code itself (``reference_arrays``), which also raises
the errors.  If the build contains something the tables cannot express, every
site takes that path.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .. import _engine
from .cone import ConeStats, build_capture_plan, build_cone, pack_cone
from .substrate import SiteCones, Substrate

_I32_MAX = 2**31 - 1


class _Unsupported(Exception):
    """The build has something the ConeCompiler would not treat as the Python code
    does: a value the integer tables cannot express, or names that make wire
    pin-node names ambiguous."""


def _fits_i32(v: int) -> bool:
    return -_I32_MAX - 1 <= v <= _I32_MAX


def _skeleton_arrays(skel, name_id) -> Dict[str, Any]:
    """The skeleton dict of load_skeleton in the CSR form of SiteCones."""
    src: List[int] = []
    topo_ptr, topo, adj_ptr, adj_u, adj_eptr, eids, dn_ptr, dn = [0], [], [0], [], [0], [], [0], []
    rows: Dict[str, int] = {}
    for site, obj in skel.items():
        rows[site] = len(src)
        src.append(name_id(site))
        topo.extend(name_id(n) for n in obj["cone_topo_nets"])
        topo_ptr.append(len(topo))
        for u, es in obj["cone_adj"]:
            adj_u.append(name_id(u))
            eids.extend(int(e) for e in es)
            adj_eptr.append(len(eids))
        adj_ptr.append(len(adj_u))
        dn.extend(name_id(n) for n in obj["reachable_dnets"])
        dn_ptr.append(len(dn))
    if not all(_fits_i32(e) for e in eids):
        raise _Unsupported("edge id beyond int32")
    i32 = lambda x: np.asarray(x, dtype=np.int32)  # noqa: E731
    i64 = lambda x: np.asarray(x, dtype=np.int64)  # noqa: E731
    return {"rows": rows, "src": i32(src), "topo_ptr": i64(topo_ptr), "topo": i32(topo),
            "adj_ptr": i64(adj_ptr), "adj_u": i32(adj_u), "adj_eptr": i64(adj_eptr), "eids": i32(eids),
            "dn_ptr": i64(dn_ptr), "dn": i32(dn)}


def build_tables(sub: Substrate) -> Dict[str, Any]:
    """The integer tables of ConeCompiler (raises _Unsupported)."""
    skel = sub.skeleton
    if isinstance(skel, SiteCones):
        names, nid = list(skel.names), dict(skel.nid)
    else:
        names, nid = [], {}

    def name_id(n: str) -> int:
        j = nid.get(n)
        if j is None:
            j = nid[n] = len(names)
            names.append(n)
        return j

    if isinstance(skel, SiteCones):
        a = skel.arrays
        sk = {"rows": None, "src": a["src"], "topo_ptr": a["topo_ptr"], "topo": a["topo"],
              "adj_ptr": a["adj_ptr"], "adj_eptr": a["adj_eptr"], "eids": a["eids"],
              "dn_ptr": a["dn_ptr"], "dn": a["dn"]}
    else:
        sk = _skeleton_arrays(skel, name_id)

    pins: Dict[str, int] = {}
    insts: Dict[str, int] = {}

    def pin_id(p: str) -> int:
        if not isinstance(p, str):
            raise _Unsupported("pin name is not a string")
        return pins.setdefault(p, len(pins))

    def inst_id(i: str) -> int:
        return insts.setdefault(i, len(insts))

    # cell entries: the dict entries of cell_arcs.json that some edge refers to
    cell_of: Dict[str, int] = {}
    ca_valid: List[int] = []
    ca_vars: List[List[int]] = []
    ca_truth: List[List[int]] = []
    ca_side: List[List[int]] = []

    def cell_id(mk: str) -> int:
        j = cell_of.get(mk)
        if j is not None:
            return j
        ent = sub.cell_arcs.get(mk)
        if not isinstance(ent, dict):
            cell_of[mk] = -1
            return -1
        vars_, truth = ent.get("vars_order"), ent.get("truth")
        valid = bool(isinstance(truth, list) and truth and isinstance(vars_, list) and vars_)
        vv: List[int] = []
        tt: List[int] = []
        if valid:
            vv = [pin_id(v) for v in vars_]
            for t in truth:
                if not isinstance(t, int) or not 0 <= int(t) <= 255:
                    raise _Unsupported("truth table value")
                tt.append(int(t))
        side = ent.get("side_pins") or []
        if not isinstance(side, list):
            raise _Unsupported("side_pins is not a list")
        j = cell_of[mk] = len(ca_valid)
        ca_valid.append(1 if valid else 0)
        ca_vars.append(vv)
        ca_truth.append(tt)
        ca_side.append([pin_id(s) for s in side])
        return j

    # edges
    edges = sub.edges
    eid_keys = list(edges)
    if eid_keys and (min(eid_keys) < 0 or max(eid_keys) > 4 * len(eid_keys) + 1_000_000):
        raise _Unsupported("edge ids are not dense")
    eid_row = np.full((max(eid_keys) + 1) if eid_keys else 0, -1, dtype=np.int32)
    gkeys: Dict[tuple, int] = {}
    cols: Dict[str, List[int]] = {k: [] for k in ("src", "dst", "gkey", "inst", "inpin", "mask", "dkey", "cause")}
    for row, eid in enumerate(eid_keys):
        e = edges[eid]
        eid_row[eid] = row
        s = name_id(e.src_net)
        cols["src"].append(s)
        cols["dst"].append(name_id(e.dst_net))
        cols["gkey"].append(gkeys.setdefault((e.inst, e.outpin, e.dst_net), len(gkeys)))
        cols["inst"].append(inst_id(e.inst))
        cols["inpin"].append(pin_id(e.inpin))
        cols["mask"].append(cell_id(e.mask_key))
        dk = int(sub.dkey2id.get(e.delay_key, -1)) if e.delay_key else -1
        if not _fits_i32(dk):
            raise _Unsupported("delay key id beyond int32")
        cols["dkey"].append(dk)
        cols["cause"].append(s if e.src_net else -1)

    # instance pins -> net (-1: unconnected, i.e. a falsy net name)
    for inst in sub.inst_pins:
        inst_id(inst)
    ip_ptr = [0]
    ip_pin: List[int] = []
    ip_nid: List[int] = []
    inst_names = list(insts)
    for inst in inst_names:
        for pin, net in sub.inst_pins.get(inst, {}).items():
            ip_pin.append(pin_id(pin))
            ip_nid.append(name_id(net) if net else -1)
        ip_ptr.append(len(ip_pin))

    # transport wires per instance pin
    has_wires = bool(sub.wire_lut)
    wl_ptr = [0]
    wl_pin: List[int] = []
    wl_rise: List[int] = []
    wl_fall: List[int] = []
    if has_wires:
        by_inst: Dict[int, List[tuple]] = {}
        for (inst, pin), (rise, fall) in sub.wire_lut.items():
            i, p = insts.get(inst), pins.get(pin)
            if i is None or p is None:
                continue                      # no gate input can match it
            by_inst.setdefault(i, []).append((p, int(rise), int(fall)))
        for i in range(len(inst_names)):
            for p, r, f in by_inst.get(i, []):
                wl_pin.append(p)
                wl_rise.append(r)
                wl_fall.append(f)
            wl_ptr.append(len(wl_pin))

    # observed FFs per D net
    ffids = sorted(sub.ffs)
    if ffids and (ffids[0] < 0 or ffids[-1] > 4 * len(ffids) + 1_000_000):
        raise _Unsupported("FF ids are not dense")
    n_ffid = (ffids[-1] + 1) if ffids else 0
    by_name: Dict[int, List[int]] = {}
    for dn, fl in sub.dnet_to_ffids.items():
        keep = [int(f) for f in fl if f in sub.ffs]
        if keep:
            by_name[name_id(dn)] = keep

    # all names are known now
    if has_wires and (any("\x01" in n for n in names) or any("/" in p for _i, p in sub.wire_lut)):
        # insert_transport_nodes names a pin node "<net>\x01<inst>/<pin>": with such
        # names two different pin nodes could share a name there
        raise _Unsupported("net or pin names that make pin-node names ambiguous")
    n_names = len(names)
    df_ptr = np.zeros(n_names + 1, dtype=np.int64)
    df_ffid: List[int] = []
    for j in range(n_names):
        df_ffid.extend(by_name.get(j, ()))
        df_ptr[j + 1] = len(df_ffid)
    gidx = np.asarray([sub.net_to_idx.get(n, -1) for n in names], dtype=np.int64)
    if gidx.size and (int(gidx.max()) > _I32_MAX):
        raise _Unsupported("net index beyond int32")
    rank = np.empty(n_names, dtype=np.int32)
    rank[np.asarray(sorted(range(n_names), key=names.__getitem__), dtype=np.int64)] = np.arange(n_names,
                                                                                                 dtype=np.int32)

    def csr(lists, dtype):
        ptr = np.zeros(len(lists) + 1, dtype=np.int64)
        np.cumsum([len(x) for x in lists], out=ptr[1:])
        flat = np.asarray([v for x in lists for v in x], dtype=dtype)
        return ptr, flat

    ca_vars_ptr, ca_vars_flat = csr(ca_vars, np.int32)
    ca_truth_ptr, ca_truth_flat = csr(ca_truth, np.uint8)
    ca_side_ptr, ca_side_flat = csr(ca_side, np.int32)
    i32 = lambda x: np.asarray(x, dtype=np.int32)  # noqa: E731
    i64 = lambda x: np.asarray(x, dtype=np.int64)  # noqa: E731
    tables = {
        "n_names": n_names, "n_gkeys": len(gkeys), "n_ffid": n_ffid, "has_wires": has_wires,
        "rank": rank, "gidx": gidx.astype(np.int32), "eid_row": eid_row,
        "e_src": i32(cols["src"]), "e_dst": i32(cols["dst"]), "e_gkey": i32(cols["gkey"]),
        "e_inst": i32(cols["inst"]), "e_inpin": i32(cols["inpin"]), "e_mask": i32(cols["mask"]),
        "e_dkey": i32(cols["dkey"]), "e_cause": i32(cols["cause"]),
        "ca_valid": np.asarray(ca_valid, dtype=np.uint8),
        "ca_vars_ptr": ca_vars_ptr, "ca_vars": ca_vars_flat,
        "ca_truth_ptr": ca_truth_ptr, "ca_truth": ca_truth_flat,
        "ca_side_ptr": ca_side_ptr, "ca_side": ca_side_flat,
        "ip_ptr": i64(ip_ptr), "ip_pin": i32(ip_pin), "ip_nid": i32(ip_nid),
        "inst_empty": np.asarray([1 if not i else 0 for i in inst_names], dtype=np.uint8),
        "wl_ptr": i64(wl_ptr), "wl_pin": i32(wl_pin), "wl_rise": i64(wl_rise), "wl_fall": i64(wl_fall),
        "df_ptr": df_ptr, "df_ffid": i32(df_ffid),
        "sk_topo_ptr": sk["topo_ptr"], "sk_topo": sk["topo"], "sk_adj_ptr": sk["adj_ptr"],
        "sk_adj_eptr": sk["adj_eptr"], "sk_eids": sk["eids"], "sk_dn_ptr": sk["dn_ptr"], "sk_dn": sk["dn"],
    }
    return {"tables": tables, "names": names, "skeleton_rows": sk["rows"], "site_src": sk["src"]}


def _compiler(sub: Substrate) -> Optional[Dict[str, Any]]:
    """ConeCompiler for this substrate (cached on it), or None if unsupported."""
    if "_fastpath_compiler" not in sub.__dict__:
        try:
            t = build_tables(sub)
        except (_Unsupported, OverflowError):       # e.g. a delay beyond int64
            t = None
        else:
            t["compiler"] = _engine.ConeCompiler(t["tables"])
        sub.__dict__["_fastpath_compiler"] = t
    return sub.__dict__["_fastpath_compiler"]


def _skeleton_row(sub: Substrate, comp: Dict[str, Any], site: str) -> Optional[int]:
    if comp["skeleton_rows"] is not None:
        return comp["skeleton_rows"].get(site)
    skel = sub.skeleton
    return skel.row(site) if site in skel else None


def reference_arrays(sub: Substrate, site: str, stats: Optional[ConeStats] = None) -> Dict[str, Any]:
    """A site compiled by the Python code, as the arrays of CompiledCone."""
    st = ConeStats() if stats is None else stats
    u0, d0 = st.unconnected_side_pins, st.dnets_without_observed_ff
    cone = build_cone(sub, site, st)
    plan = build_capture_plan(sub, cone, st)
    gidx = np.asarray([sub.net_to_idx.get(n, -1) for n in cone.i2net], dtype=np.int32)
    dp_g = (np.asarray([sub.net_to_idx.get(dn, -1) for dn in plan.dp_dnet_names], dtype=np.int64)
            if plan.dp_dnet_names else np.zeros(0, np.int64))
    packed = pack_cone(cone, sub.dkey2id)
    src_i = cone.net2i[site]
    settle_gate: List[int] = []
    settle_in: List[int] = []
    settle_in_ptr = [0]
    for gi, g in enumerate(cone.gates):
        if gidx[g.out_i] < 0:
            settle_gate.append(gi)
            settle_in.extend(int(x) for x in (g.var_nets_src or g.var_nets))
            settle_in_ptr.append(len(settle_in))
    out = {f: getattr(packed, f) for f in (
        "fanout_ptr", "fanout_idx", "gate_out", "gate_var_ptr", "gate_var_nets", "gate_truth_ptr",
        "gate_truth", "arc_dkey_id", "arc_cause_i", "arc_side_mode", "arc_side_ptr", "arc_side_nets",
        "rec_idx", "wire_ptr", "wire_zsplit", "wire_sink", "wire_rise_ps", "wire_fall_ps")}
    out.update({f: getattr(plan, f) for f in ("ffids", "dp_rec_slot", "dp_ptr", "dp_idx")})
    out.update(src_i=int(src_i), gidx=gidx, dp_g=dp_g,
               settle_gate=np.asarray(settle_gate, dtype=np.int32),
               settle_in_ptr=np.asarray(settle_in_ptr, dtype=np.int32),
               settle_in=np.asarray(settle_in, dtype=np.int32),
               wp_drv=np.asarray([w[0] for w in cone.wire_pairs], dtype=np.int32),
               wp_sink=np.asarray([w[1] for w in cone.wire_pairs], dtype=np.int32),
               unconnected_side_pins=st.unconnected_side_pins - u0,
               dnets_without_observed_ff=st.dnets_without_observed_ff - d0,
               i2net=list(cone.i2net), dp_dnet_names=list(plan.dp_dnet_names))
    return out


def reference_cone(sub: Substrate, site: str):
    arrays = reference_arrays(sub, site)
    arrays.pop("i2net")
    arrays.pop("dp_dnet_names")
    return _engine.CompiledCone.from_arrays(arrays)


def compile_sites(sub: Substrate, sites: Sequence[str], n_threads: int = 1,
                  keep_names: bool = False) -> List[Any]:
    """One CompiledCone per site, in order."""
    out: List[Any] = [None] * len(sites)
    comp = _compiler(sub)
    if comp is not None:
        idx, rows = [], []
        for i, s in enumerate(sites):
            r = _skeleton_row(sub, comp, s)
            if r is not None:
                idx.append(i)
                rows.append(r)
        if idx:
            rows_a = np.asarray(rows, dtype=np.int64)
            cones, _status = comp["compiler"].compile(rows_a, comp["site_src"][rows_a].astype(np.int32),
                                                      int(n_threads), bool(keep_names))
            for i, c in zip(idx, cones):
                out[i] = c
    for i, s in enumerate(sites):
        if out[i] is None:            # not handled by the compiler: the Python code (and its errors)
            out[i] = reference_cone(sub, s)
    return out


def compiled_names(sub: Substrate) -> Optional[List[str]]:
    """Name of each name id of the compiler (to read CompiledCone.arrays()['net_nid'])."""
    comp = _compiler(sub)
    return None if comp is None else comp["names"]
