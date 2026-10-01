"""``run_build``: netlist + SDF + behavioural library -> one substrate directory."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from . import checks, graph, netlist, observe, timing
from .celllib import CellFunctions, check_used_cells, is_structurally_untimed, parse_library
from .spec import BuildError, BuildSpec

log = logging.getLogger("setfi.build")

MANIFEST_NAME = "build_manifest.json"
INTERMEDIATE_DIR = "intermediate"

# What the campaign reads.  interconnect is written only when the SDF has INTERCONNECT records.
CONTRACT_FILES = (
    "net_index.json",
    "site_cones.jsonl",
    "edges.json",
    "pin_to_net.json",
    "ff_index.json",
    "cell_arcs.json",
    "timing_checks.json",
    "delay_table.npz",
    "arc_index.json",
    "interconnect.json",
)

MANIFEST_FORMAT = 1


def _file_info(path: str) -> Dict[str, Any]:
    h = hashlib.md5()
    n = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
    return {"path": os.path.abspath(path), "md5": h.hexdigest(), "bytes": n}


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _dump(obj: Any, path: str, **kw: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, **kw)


def _full_match_any(patterns):
    compiled = [re.compile(p) for p in patterns]

    def f(name: str) -> bool:
        return any(r.fullmatch(name) for r in compiled)
    return f


def run_build(spec: BuildSpec) -> Dict[str, Any]:
    """Build the timing graph described by ``spec`` into ``spec.out_dir``.

    Writes the contract files (see ``CONTRACT_FILES``) and ``build_manifest.json``,
    and returns the manifest.  Raises :class:`BuildError` when the inputs cannot be
    turned into a trustworthy graph (unknown cells, cells without a parsable function,
    netlist/SDF disagreement, parse-integrity failures, cone self-check mismatch, ...).
    """
    spec.validate()
    out_dir = os.path.abspath(spec.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    # A previous build's files must not survive into this one (e.g. interconnect.json next
    # to a build whose SDF has no INTERCONNECT), and a failed build must not leave
    # half-written files that look loadable.
    _remove_outputs(out_dir)
    try:
        return _build(spec, out_dir)
    except BaseException:
        _remove_outputs(out_dir)
        raise


def _remove_outputs(out_dir: str) -> None:
    for name in CONTRACT_FILES + (MANIFEST_NAME,):
        p = os.path.join(out_dir, name)
        if os.path.lexists(p):
            os.remove(p)


def _build(spec: BuildSpec, out_dir: str) -> Dict[str, Any]:
    t0 = time.time()
    lspec = spec.library
    warnings: List[str] = []
    inter_dir: Optional[str] = None
    if spec.keep_intermediates:
        inter_dir = os.path.join(out_dir, INTERMEDIATE_DIR)
        os.makedirs(inter_dir, exist_ok=True)

    inputs = {
        "netlist": _file_info(spec.netlist),
        "sdf": _file_info(spec.sdf),
        "cell_functions": _file_info(lspec.behavioral_verilog),
    }

    # ---- library ----
    cells = parse_library(_read(lspec.behavioral_verilog), lspec)
    lib_celltypes = set(cells)
    log.info("library: %d cells (%d sequential)", len(cells),
             sum(c.is_sequential for c in cells.values()))

    # ---- SDF ----
    nl_text = _read(spec.netlist)
    mods = re.findall(r"^\s*module\s+([A-Za-z_][A-Za-z0-9_$]*)", netlist.strip_comments(nl_text), re.M)
    if spec.top not in mods:
        raise BuildError(f"design.top = {spec.top!r}: no module of that name in design.netlist "
                         f"(its modules: {', '.join(sorted(mods)[:10])}"
                         f"{', ...' if len(mods) > 10 else ''})")
    sdf = timing.read_sdf(spec.sdf)
    design = sdf.header.get("DESIGN", "")
    # The SDF must have been written for this top: an SDF of another design would
    # annotate little or nothing.
    if design != spec.top:
        raise BuildError(f"design.sdf was written for the design {design!r} (its DESIGN "
                         f"entry), but design.top is {spec.top!r}; the SDF must belong to "
                         f"this netlist")
    log.info("SDF: %s %s, %d arcs, %d timing checks, %d interconnect rows",
             sdf.header.get("PROGRAM", "?"), sdf.header.get("SDFVERSION", "?"),
             len(sdf.arcs), len(sdf.tcs), len(sdf.ics))

    # ---- netlist ----
    esc = re.search(r"\\\S+", netlist.strip_comments(nl_text))
    if esc:
        line = netlist.strip_comments(nl_text)[:esc.start()].count("\n") + 1
        raise BuildError(f"{spec.netlist}:{line}: escaped identifier {esc.group(0)!r}; escaped "
                         f"names are not supported (write the netlist without them)")
    modules = netlist.parse_modules(nl_text)
    is_blackbox = _full_match_any(lspec.blackbox_cells)
    elab = netlist.elaborate(modules, spec.top, lib_celltypes, is_blackbox)
    insts = elab.leaves
    for path, inst in insts.items():
        c = cells.get(inst.celltype)
        if c is None:
            continue
        pins = set(c.inputs) | set(c.outputs) | set(c.inouts)
        bad = sorted(set(inst.pin2net) - pins)
        if bad:
            raise BuildError(f"instance {path} ({inst.celltype}) connects pin(s) {bad} that the "
                             f"cell does not have (cell pins: {sorted(pins)})")
    log.info("netlist: %d modules, %d instances, %d black boxes", len(modules), len(insts),
             len(elab.blackboxes))
    if elab.n_assign:
        log.info("netlist: %d assign statement(s): %d bit(s) merged as aliases, %d tied to "
                 "constants", elab.n_assign, elab.n_alias_bits, elab.n_tied_bits)

    user_untimed = _full_match_any(lspec.untimed_cells)

    def is_untimed(celltype: str) -> bool:
        c = cells.get(celltype)
        return (c is not None and is_structurally_untimed(c)) or user_untimed(celltype)

    xv = netlist.crosscheck_sdf(insts, elab.blackboxes, sdf.inst_celltype, lib_celltypes,
                                is_untimed)
    if not xv["ok"]:
        parts = []
        if xv["extra"]:
            parts.append(f"{len(xv['extra'])} netlist instance(s) have no SDF entry, e.g. "
                         f"{xv['extra'][:5]} (add them to the SDF, or list cells without "
                         f"delays in library.untimed_cells)")
        if xv["missing"]:
            parts.append(f"{len(xv['missing'])} SDF instance(s) are not in the netlist, e.g. "
                         f"{xv['missing'][:5]}")
        if xv["celltype_mismatch"]:
            ex = [f"{p}: netlist {insts[p].celltype}, SDF {sdf.inst_celltype[p]}"
                  for p in xv["celltype_mismatch"][:3]]
            parts.append(f"{len(xv['celltype_mismatch'])} instance(s) have a different cell "
                         f"type in the SDF: {ex}")
        msg = ("the netlist and the SDF disagree: " + "; ".join(parts)
               + " (build.strict_sdf_crosscheck = false accepts this)")
        if spec.strict_sdf_crosscheck:
            raise BuildError(msg)
        warnings.append(msg)

    used = {inst.celltype for inst in insts.values() if inst.celltype in cells}
    problems = check_used_cells(cells, used)
    if problems:
        raise BuildError(f"{len(problems)} library cell(s) used by the netlist have no "
                         f"usable model:\n  " + "\n  ".join(problems))

    primary_inputs = netlist.primary_input_bits(modules[spec.top])

    # ---- graph ----
    g = graph.build_stage(
        insts, cells, primary_inputs,
        os.path.join(out_dir, "site_cones.jsonl"),
        exclude_q_from_src=spec.exclude_q_from_src,
        submodules=set(modules) - set(cells),
        exclude_d_from_src=spec.exclude_d_from_src,
        self_check_samples=spec.self_check_samples, self_check_seed=spec.self_check_seed,
        do_self_check=spec.self_check)
    log.info("graph: %d nets, %d edges, %d sites, %d FFs", len(g.nets), len(g.edges),
             len(g.src_list), len(g.ff_index["ffid_to_info"]))
    if g.counts.get("n_sites_driven_by_nothing"):
        warnings.append(f"{g.counts['n_sites_driven_by_nothing']} injection site(s) "
                        f"are driven by no gate, flip-flop, primary input or black box")

    # ---- parse integrity (absolute checks on the parsed netlist) ----
    integ = checks.parse_integrity(g.idx_to_net.values(), modules, spec.top, cells,
                                   spec.max_undriven_nets, aliases=elab.aliases)
    if integ["failures"]:
        raise BuildError("the netlist check failed:\n  " + "\n  ".join(integ["failures"]))

    _dump({"net_to_idx": g.net_to_idx,
           "idx_to_net": {str(i): n for i, n in g.idx_to_net.items()}},
          os.path.join(out_dir, "net_index.json"), ensure_ascii=False)
    _dump(g.pin_to_net, os.path.join(out_dir, "pin_to_net.json"), ensure_ascii=False)
    _dump([{**e.__dict__, "src_net_idx": g.net_to_idx[e.src_net],
            "dst_net_idx": g.net_to_idx[e.dst_net]} for e in g.edges],
          os.path.join(out_dir, "edges.json"), ensure_ascii=False)

    # ---- timing ----
    funcs = CellFunctions(cells, spec.max_enum_side_pins, lspec.scan_enable_pin_names)
    opts = timing.TimingOptions(
        rail=spec.sdf_rail, time_precision_ps=spec.time_precision_ps,
        side_pins_topk=spec.side_pins_topk, exclude_src_from_side=spec.exclude_src_from_side,
        mirror_scan_data_checks=lspec.mirror_scan_data_checks,
        scan_data_pin_pairs=lspec.scan_data_pin_pairs)
    tres = timing.compile_stage(sdf, funcs, g.inst_to_idx, g.net_to_idx, g.pin_to_net,
                                out_dir, opts, intermediate_dir=inter_dir)
    log.info("timing: %d arcs, %d checks, %d interconnect rows", tres["n_arcs"],
             tres["n_timing_checks"], tres["n_interconnect"])

    # every gate arc a pulse can traverse needs an SDF delay: without one the pulse
    # cannot pass that arc at all
    with open(os.path.join(out_dir, "arc_index.json")) as fh:
        arc_keys = set(json.load(fh).get("delay_key_to_arc_idx", {}))
    no_delay: Dict[Tuple[str, str, str], List[str]] = {}
    for eid in sorted(g.cone_edge_ids):
        e = g.edges[eid]
        if e.delay_key not in arc_keys:
            no_delay.setdefault((e.celltype, e.inpin, e.outpin), []).append(e.inst)
    if no_delay:
        items = sorted(no_delay.items(), key=lambda kv: -len(kv[1]))
        desc = "; ".join(f"{ct} {a}->{z} ({len(insts)} instance(s), e.g. {insts[0]})"
                         for (ct, a, z), insts in items[:5])
        msg = (f"{sum(len(v) for v in no_delay.values())} gate arc(s) inside injection cones "
               f"have no SDF IOPATH delay: {desc}")
        if spec.allow_missing_delays:
            warnings.append(msg + "; pulses cannot pass these arcs")
        else:
            raise BuildError(msg + ". Add them to the SDF, or set build.allow_missing_delays "
                             "= true to let pulses stop at these arcs.")

    # ---- observed FFs ----
    ffx = observe.filter_observed(g.ff_index, lspec.non_observed_cells,
                                  lspec.observed_d_pins, spec.non_observed_instances)
    _dump(ffx, os.path.join(out_dir, "ff_index.json"), indent=2)
    obs = dict(ffx["_meta"]["observe_filter"])
    obs["removed_examples"] = observe.removed_examples(g.ff_index, ffx)

    # ---- intermediates ----
    if inter_dir:
        _write_intermediates(inter_dir, g)

    manifest: Dict[str, Any] = {
        "tool": "setfi.build",
        "format": MANIFEST_FORMAT,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": inputs,
        "top": spec.top,
        "sdf_header": sdf.header,
        "sdf_reader_counts": dict(sdf.notes),
        "artifacts": [n for n in CONTRACT_FILES if os.path.exists(os.path.join(out_dir, n))],
        "netlist": {
            "n_modules": len(modules),
            "n_instances": sum(1 for i in insts.values() if i.celltype not in modules),
            "n_blackboxes": len(elab.blackboxes),
            "n_assign": elab.n_assign,
            "n_assign_alias_bits": elab.n_alias_bits,
            "n_assign_tied_bits": elab.n_tied_bits,
            "sdf_crosscheck": {k: (v[:20] if isinstance(v, list) else v)
                               for k, v in xv.items()},
        },
        "library": {
            "n_cells": len(cells),
            "n_used_cell_types": len(used),
            "used_sequential": sorted(c for c in used if cells[c].is_sequential),
        },
        "graph": g.counts,
        "timing": {k: v for k, v in tres.items() if k != "diagnostics"},
        "observed_ffs": obs,
        "checks": {"parse_integrity": integ,
                   "cone_self_check_samples_ok": g.counts.get("self_check_samples_ok", 0),
                   "side_pin_bases_checked": tres["n_side_pin_bases_checked"],
                   "cell_models": "ok"},
        "sdf_notes": {
            "about": "SDF entries by why they are not used to propagate pulses; e.g. a "
                     "flip-flop's clock-to-output arc is not combinational. Arcs inside "
                     "injection cones are checked separately (a missing delay fails the build).",
            **tres["diagnostics"]},
        "warnings": warnings,
        "elapsed_s": round(time.time() - t0, 3),
    }
    _dump(manifest, os.path.join(out_dir, MANIFEST_NAME), indent=2, ensure_ascii=False)
    for w in warnings:
        log.warning("%s", w)
    return manifest


def _write_intermediates(d: str, g: "graph.GraphResult") -> None:
    """Products that only explain the contract files (not read by the campaign)."""
    with open(os.path.join(d, "cone_db.jsonl"), "w", encoding="utf-8") as f:
        for u in g.conedb_order:
            f.write(json.dumps({"keep_net": u,
                                "cone_seid_blocks": g.conedb[u].to_blocks_list()},
                               ensure_ascii=False) + "\n")
    _dump([se.__dict__ for se in g.superedges], os.path.join(d, "superedges.json"),
          ensure_ascii=False)
    _dump(g.net_entry, os.path.join(d, "net_entry.json"), ensure_ascii=False)
    _dump({"version": 1, "n_insts": len(g.idx_to_inst), "inst_to_idx": g.inst_to_idx,
           "idx_to_inst": g.idx_to_inst}, os.path.join(d, "inst_index.json"),
          ensure_ascii=False)
    _dump({"version": 1, "n_sites": len(g.src_list),
           "site_net_idxs": [g.net_to_idx[s] for s in g.src_list]},
          os.path.join(d, "site_index.json"), ensure_ascii=False)
    _dump({k: g.net_to_idx[v] for k, v in g.pin_to_net.items() if v in g.net_to_idx},
          os.path.join(d, "pin_to_net_idx.json"), ensure_ascii=False)
    _dump({"nets": sorted(g.non_lib_connected_nets)},
          os.path.join(d, "non_lib_connected_nets.json"), indent=2)
