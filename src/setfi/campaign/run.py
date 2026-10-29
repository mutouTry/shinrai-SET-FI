"""The injection campaign: every (sampled cycle, site, pulse width) trial.

For each trial the SET is injected on the site's net at t = 0 of a simulation
that starts from the recorded state of that cycle, propagated through the
site's fan-out cone, and every pulse that reaches the D pin of an observed FF
is written as a record (see ``setfi.records``).  The SET start time within the
cycle is not simulated separately: it only shifts the pulse, and the capture
condition is applied analytically by the analysis step.

The trials run in C++ (``_engine.run_block``, one call per block of cycles,
``jobs`` threads) on cones compiled in C++ (``fastpath.compile_sites``), after
site_cones.jsonl has been read in C++.  The Python code that does the same (the
site_cones.jsonl reader, ``cone.build_cone`` and the ``_Worker`` loop on ``jobs``
processes) is kept as the reference engine, against which the tests check the
C++ code: set SETFI_REFERENCE_ENGINE=1 to use it.  Both write the same files.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

import numpy as np

from .. import _engine
from ..records import COLUMNS, FORMAT, INF_PS, WINDOWS_FILE, write_shard, write_windows
from .cone import ConeStats, CapturePlan, PackedCone, build_capture_plan, build_cone, pack_cone
from .fastpath import compile_sites
from .state import read_cycle_bits
from .substrate import SiteCones, Substrate, cone_arcs_without_delay, load_substrate
from .windows import campaign_windows

STAT_KEYS = ("applied_events", "canceled_replace", "canceled_back_to_current", "kept_same_target",
             "coalesced_nets", "coalesced_events", "dropped_arc_no_delay", "dropped_no_candidate_arc")
# columns of run_block's site_stats
SITE_STATS = ("n_sims",) + STAT_KEYS + ("max_heap", "n_sims_reaching_d", "n_sims_with_records")


@dataclass
class CampaignSpec:
    stage_dir: str                    # build output
    state_dir: str                    # cycle_<N>.hex files
    cycles: List[int]                 # sampled cycles, in campaign order
    out_dir: str
    clock_period_ps: int
    pulse_widths_ps: List[int]
    capture_edge_ps: Optional[int] = None     # default: clock period
    sim_horizon_ps: Optional[int] = None      # default: 1.25 x capture edge
    delay_choice: str = "max"                 # "max" | "min" when no changed input has an active path
    vcs_align: bool = False                   # propagation.vcs_align
    coalesce_window_ps: int = 0
    interconnect: str = "ignore"              # "ignore" | "transport"
    clock_pins: List[str] = field(default_factory=lambda: ["CP", "CK", "CLK"])
    allow_missing_timing_checks: bool = False
    sites: Optional[List[str]] = None         # subset of injection sites (default: all)
    cycles_per_shard: int = 10
    jobs: int = 1
    timeprecision_ps: int = 1

    def resolved(self) -> "CampaignSpec":
        s = CampaignSpec(**asdict(self))
        if s.capture_edge_ps is None:
            s.capture_edge_ps = int(s.clock_period_ps)
        if s.sim_horizon_ps is None:
            s.sim_horizon_ps = int(Fraction(5, 4) * s.capture_edge_ps)
        return s


def engine_options(spec: CampaignSpec) -> Dict[str, object]:
    """The propagation flags of a (resolved) spec, as run_block takes them.

    simulate_pulses takes the same flags as positional arguments, in this order."""
    return {"sim_end_ps": int(spec.sim_horizon_ps), "eps_ps": int(spec.coalesce_window_ps),
            "delay_policy_is_max": spec.delay_choice == "max", "vcs_align": bool(spec.vcs_align)}


@dataclass
class CompiledSite:
    """Everything the trial loop needs for one site; the full cone is not kept."""
    packed: PackedCone
    plan: CapturePlan
    gidx: np.ndarray                  # cone-local net -> global net index (-1: not recorded)
    src_i: int
    wire_pairs: List[Tuple[int, int, int, int]]
    settle: Optional[List[Tuple[int, List[int], List[int]]]]   # gates whose output is not recorded
    dp_g: np.ndarray                  # global net index of each recorded D net


class _Worker:
    """The reference engine's trial loop.  One process: a fixed slice of the
    injection sites, all cycles."""

    def __init__(self, spec: CampaignSpec, sub: Substrate, sites: List[str], site_index: Dict[str, int]):
        self.spec = spec
        self.sub = sub
        self.sites = sites
        self.site_idx = site_index
        self.cone_stats = ConeStats()
        self.cones: Dict[str, CompiledSite] = {}
        self.pws = np.asarray(spec.pulse_widths_ps, dtype=np.int64)

    def cone(self, site: str) -> CompiledSite:
        c = self.cones.get(site)
        if c is None:
            sub = self.sub
            cone = build_cone(sub, site, self.cone_stats)
            plan = build_capture_plan(sub, cone, self.cone_stats)
            gidx = np.asarray([sub.net_to_idx.get(n, -1) for n in cone.i2net], dtype=np.int32)
            valid = gidx >= 0
            settle = None
            if not valid.all():
                settle = [(g.out_i, list(g.var_nets_src or g.var_nets), list(g.truth))
                          for g in cone.gates if not valid[g.out_i]]
            dp_g = (np.asarray([sub.net_to_idx.get(dn, -1) for dn in plan.dp_dnet_names], dtype=np.int64)
                    if plan.dp_dnet_names else np.zeros(0, np.int64))
            c = CompiledSite(packed=pack_cone(cone, sub.dkey2id), plan=plan, gidx=gidx,
                             src_i=cone.net2i[site], wire_pairs=list(cone.wire_pairs),
                             settle=settle, dp_g=dp_g)
            self.cones[site] = c
        return c

    def run_block(self, shard_id: str, cycles: List[int]) -> Dict:
        spec = self.spec
        sub = self.sub
        rec = {k: [] for k in ("cycle", "site", "width_ps", "ff", "base", "t_enter", "t_exit")}
        stats = {k: 0 for k in STAT_KEYS}
        stats.update(n_sims=0, n_sims_reaching_d=0, n_sims_with_records=0, max_heap=0)
        n_nets = len(sub.idx_to_net)
        for cyc in cycles:
            bits = read_cycle_bits(spec.state_dir, cyc, n_nets)
            for site in self.sites:
                cs = self.cone(site)
                plan = cs.plan
                gidx = cs.gidx
                valid = gidx >= 0
                net_val = np.where(valid, bits[np.where(valid, gidx, 0)], 0).astype(np.uint8)
                if cs.settle:                      # nets outside the recording: settle
                    for out_i, ins, truth in cs.settle:
                        idx = 0
                        for j, ni in enumerate(ins):
                            idx |= (int(net_val[ni]) & 1) << j
                        net_val[out_i] = (truth[idx] if len(truth) > 1 else truth[0]) & 1
                for drv, sink, _r, _f in cs.wire_pairs:
                    net_val[sink] = net_val[drv] & 1
                v0 = int(net_val[cs.src_i]) & 1
                dp_g = cs.dp_g
                dp_base = np.where(dp_g >= 0, bits[np.where(dp_g >= 0, dp_g, 0)], 0).astype(np.uint8)

                results = _engine.simulate_pulses(
                    cs.packed, sub.delaytab, net_val, int(cs.src_i), v0, v0 ^ 1, self.pws,
                    int(spec.sim_horizon_ps), int(spec.coalesce_window_ps),
                    spec.delay_choice == "max", bool(spec.vcs_align))
                sidx = self.site_idx[site]
                for pw, (ev_ptr, ev_t, ev_v, st) in zip(self.pws.tolist(), results):
                    stats["n_sims"] += 1
                    for k in STAT_KEYS:
                        stats[k] += int(st[k])
                    stats["max_heap"] = max(stats["max_heap"], int(st["max_heap"]))
                    if plan.n_ff == 0 or plan.dp_rec_slot.size == 0:
                        continue
                    if ev_ptr.size == 0 or int(ev_ptr[-1]) == 0:
                        continue
                    stats["n_sims_reaching_d"] += 1
                    dpi, t_a, t_b = _engine.extract_pulses(ev_ptr, ev_t, ev_v, plan.dp_rec_slot,
                                                           dp_base, INF_PS)
                    if dpi.size:
                        counts = plan.dp_ptr[dpi + 1] - plan.dp_ptr[dpi]
                        total = int(counts.sum())
                        if total:
                            stats["n_sims_with_records"] += 1
                            first = np.repeat(plan.dp_ptr[dpi], counts)
                            offs = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
                            li = plan.dp_idx[first + offs]
                            rep = np.repeat(np.arange(dpi.size), counts)
                            rec["cycle"].append(np.full(total, cyc, np.int32))
                            rec["site"].append(np.full(total, sidx, np.int32))
                            rec["width_ps"].append(np.full(total, pw, np.int32))
                            rec["ff"].append(plan.ffids[li])
                            rec["base"].append(dp_base[dpi][rep])
                            rec["t_enter"].append(t_a[rep])
                            rec["t_exit"].append(t_b[rep])

        cols = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in rec.items()}
        shard = write_shard(os.path.join(spec.out_dir, f"records_{shard_id}.npz"), cols)
        shard.update(cycles=[int(c) for c in cycles])
        stats["unconnected_side_pins"] = self.cone_stats.unconnected_side_pins
        return {"shard": shard, "stats": stats}

    def run_all(self, blocks, chunk: int, n_chunks: int, log=None) -> List[Dict]:
        out = []
        for b, cycles in blocks:
            sid = f"c{b:03d}_s{chunk:03d}"
            r = self.run_block(sid, cycles)
            r["site_slice"] = chunk
            out.append(r)
            if log:
                log(f"[inject] {r['shard']['file']}: {r['shard']['n_records']:,} records")
        return out


def _write_slice(spec: CampaignSpec, b: int, chunk: int, cycles: List[int], out: Dict, lo: int, hi: int,
                 n_sites: int, unconnected: np.ndarray) -> Dict:
    """The record file of block b, site slice chunk = sites[lo:hi]."""
    sid = f"c{b:03d}_s{chunk:03d}"
    off = out["rec_off"]
    spans = [(int(off[k * n_sites + lo]), int(off[k * n_sites + hi])) for k in range(len(cycles))]
    cols = {k: np.concatenate([out[k][a:z] for a, z in spans]) if spans else out[k][:0] for k in COLUMNS}
    shard = write_shard(os.path.join(spec.out_dir, f"records_{sid}.npz"), cols)
    shard.update(cycles=[int(c) for c in cycles])
    S = out["site_stats"][lo:hi]
    col = {k: S[:, i] for i, k in enumerate(SITE_STATS)}
    stats = {k: int(col[k].sum()) for k in STAT_KEYS}
    stats.update(n_sims=int(col["n_sims"].sum()), n_sims_reaching_d=int(col["n_sims_reaching_d"].sum()),
                 n_sims_with_records=int(col["n_sims_with_records"].sum()),
                 max_heap=int(col["max_heap"].max()) if hi > lo else 0)
    stats["unconnected_side_pins"] = int(unconnected[lo:hi].sum())
    return {"shard": shard, "stats": stats, "site_slice": chunk}


def _run_fast(spec: CampaignSpec, sub: Substrate, sites: List[str], site_index: Dict[str, int],
              blocks, bounds: List[int], log, timing: Dict[str, float]) -> List[Dict]:
    """All blocks with run_block; results in the order of the reference engine."""
    n_chunks = len(bounds) - 1
    if not blocks:               # no cycles: the reference engine never compiles a cone either
        return []
    pws = np.asarray(spec.pulse_widths_ps, dtype=np.int64)
    t = time.time()
    cones = compile_sites(sub, sites, n_threads=n_chunks)
    timing["compile"] = time.time() - t
    site_ids = np.asarray([site_index[s] for s in sites], dtype=np.int32)
    unconnected = np.asarray([c.unconnected_side_pins for c in cones], dtype=np.int64)
    opts = engine_options(spec)
    n_nets = len(sub.idx_to_net)
    by_chunk: List[List[Dict]] = [[] for _ in range(n_chunks)]
    with ThreadPoolExecutor(max_workers=n_chunks) as pool:
        for b, cycles in blocks:
            t = time.time()
            bits = np.stack([read_cycle_bits(spec.state_dir, c, n_nets) for c in cycles])
            out = _engine.run_block(cones, bits, np.asarray(cycles, dtype=np.int64), site_ids, pws,
                                    sub.delaytab, opts, INF_PS, n_chunks)
            del bits
            t2 = time.time()
            timing["trials"] = timing.get("trials", 0.0) + (t2 - t)
            futures = [pool.submit(_write_slice, spec, b, ch, cycles, out, bounds[ch], bounds[ch + 1],
                                   len(sites), unconnected) for ch in range(n_chunks)]
            for ch, fu in enumerate(futures):
                r = fu.result()
                by_chunk[ch].append(r)
                log(f"[inject] {r['shard']['file']}: {r['shard']['n_records']:,} records")
            del out
            timing["write"] = timing.get("write", 0.0) + (time.time() - t2)
    return [r for rs in by_chunk for r in rs]


def _reference_engine() -> bool:
    return os.environ.get("SETFI_REFERENCE_ENGINE", "") not in ("", "0")


_G: Dict[str, object] = {}


def _run_slice_in_child(chunk: int):
    spec, sub, slices, site_index, blocks = (_G[k] for k in ("spec", "sub", "slices", "site_index", "blocks"))
    w = _Worker(spec, sub, slices[chunk], site_index)
    return w.run_all(blocks, chunk, len(slices))


def check_sites(sites: List[str], stage_dir: str) -> None:
    """Refuse site names that are not injection sites, before anything is deleted."""
    if not sites:
        return
    from .substrate import load_net_index, load_site_cones
    net_to_idx, idx_to_net = load_net_index(os.path.join(stage_dir, "net_index.json"))
    fast = None if _reference_engine() else load_site_cones(os.path.join(stage_dir, "site_cones.jsonl"),
                                                            idx_to_net)
    if fast is not None:
        names = set(fast)
    else:
        site_idx = set()
        with open(os.path.join(stage_dir, "site_cones.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    site_idx.add(int(json.loads(line)["src_net_idx"]))
        names = {idx_to_net[i] for i in site_idx}
    _refuse_unknown_sites(sites, names)


def select_sites(sub: Substrate, sites: Optional[List[str]], stage_dir: str) -> List[str]:
    """The injection sites of a campaign, sorted: `sites` (refused if one is not an
    injection site), or every site if `sites` is empty."""
    if not sites:
        return sorted(sub.skeleton)
    if isinstance(sub.skeleton, SiteCones):    # the same names check_sites reads
        _refuse_unknown_sites(list(sites), set(sub.skeleton))
    else:
        check_sites(list(sites), stage_dir)
    return sorted(set(sites))


def _refuse_unknown_sites(sites: List[str], names) -> None:
    import difflib
    from .substrate import SubstrateError
    bad = [s for s in sites if s not in names]
    if bad:
        hint = difflib.get_close_matches(bad[0], sorted(names), n=3)
        raise SubstrateError(
            f"fault_model.sites: {len(bad)} name(s) are not injection sites, e.g. {bad[0]!r}"
            + (f" (close matches: {hint})" if hint else "")
            + ". Sites are net names with `/` between hierarchy levels; a net connected to "
              "a module port is named after the net in the enclosing module.")


def run_campaign(spec: CampaignSpec, log=print) -> Dict:
    spec = spec.resolved()
    reference = _reference_engine()
    t0 = time.time()
    sub = load_substrate(spec.stage_dir, clock_pins=spec.clock_pins, interconnect=spec.interconnect,
                         timeprecision_ps=spec.timeprecision_ps,
                         allow_missing_timing_checks=spec.allow_missing_timing_checks,
                         n_threads=None if reference else max(1, int(spec.jobs)))
    timing = {"load": time.time() - t0}
    sites = select_sites(sub, spec.sites, spec.stage_dir)
    site_index = {s: i for i, s in enumerate(sites)}
    os.makedirs(spec.out_dir, exist_ok=True)
    for f in os.listdir(spec.out_dir):          # a previous campaign's files must not survive
        if f.startswith("records_") or f in ("index.json", WINDOWS_FILE):
            os.remove(os.path.join(spec.out_dir, f))
    log(f"[inject] {len(spec.cycles)} cycles x {len(sites)} sites x {len(spec.pulse_widths_ps)} widths "
        f"= {len(spec.cycles) * len(sites) * len(spec.pulse_widths_ps):,} trials; "
        f"{sub.counts['n_ff_observed']} observed FFs; T = {spec.clock_period_ps} ps")
    n_no_delay, n_no_delay_sites = cone_arcs_without_delay(sub, sites)
    if n_no_delay:
        log(f"[inject] {n_no_delay} gate arc(s) in the cones of {n_no_delay_sites} site(s) have no "
            f"delay (build.allow_missing_delays): pulses do not pass them")

    blocks = [(b, spec.cycles[i:i + spec.cycles_per_shard])
              for b, i in enumerate(range(0, len(spec.cycles), spec.cycles_per_shard))]
    # The sites are cut into n_chunks contiguous slices, one record file per (block,
    # slice).  The C++ engine shares the sites of a block out over n_chunks threads;
    # the reference engine runs one process per slice, which builds (and holds) only
    # the cones of its own sites.
    n_chunks = max(1, min(int(spec.jobs), len(sites)))
    bounds = [round(i * len(sites) / n_chunks) for i in range(n_chunks + 1)]
    slices = [sites[bounds[i]:bounds[i + 1]] for i in range(n_chunks)]
    results: List[Dict] = []
    if not reference:
        results = _run_fast(spec, sub, sites, site_index, blocks, bounds, log, timing)
    elif n_chunks > 1:
        _G.update(spec=spec, sub=sub, slices=slices, site_index=site_index, blocks=blocks)
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=n_chunks) as pool:
            for chunk_results in pool.imap(_run_slice_in_child, range(n_chunks)):
                results.extend(chunk_results)
                log(f"[inject] site slice {chunk_results[0]['site_slice'] + 1 if chunk_results else '?'}"
                    f"/{n_chunks} done: {sum(r['shard']['n_records'] for r in chunk_results):,} records")
        _G.clear()
    else:
        results = _Worker(spec, sub, sites, site_index).run_all(blocks, 0, 1, log)

    stats: Dict[str, int] = {}
    for r in results:
        for k, v in r["stats"].items():
            stats[k] = max(stats.get(k, 0), v) if k in ("max_heap", "unconnected_side_pins") else stats.get(k, 0) + v
    counts = {"trials": stats.get("n_sims", 0),
              "trials_reaching_an_observed_ff": stats.get("n_sims_with_records", 0),
              "records": sum(r["shard"]["n_records"] for r in results)}
    if n_no_delay:
        counts["arcs_without_delay"] = n_no_delay

    win = campaign_windows(sub, spec.stage_dir, spec.state_dir, spec.cycles, spec.clock_pins,
                           spec.capture_edge_ps, spec.timeprecision_ps)
    windows = write_windows(os.path.join(spec.out_dir, WINDOWS_FILE), win.exc_cycle, win.exc_ff,
                            win.exc_enter, win.exc_leave)
    if win.n_unread_checks:
        log(f"[inject] {win.n_unread_checks} setup/hold check(s) have a COND that is not made of "
            f"pin == value terms; they use their first SDF entry in every cycle")
    T = int(spec.capture_edge_ps)
    ffs = []
    for ffid, (e0, e1, l0, l1) in zip(win.ffids.tolist(), win.default.tolist()):
        ff = sub.ffs[ffid]
        ffs.append({"id": ffid, "inst": ff.inst, "d_net": ff.d_net, "d_pin": ff.d_pin,
                    "setup_rise_ps": T - l1, "hold_rise_ps": e0 - T,
                    "setup_fall_ps": T - l0, "hold_fall_ps": e1 - T,
                    "capture_enter_ps": [e0, e1], "capture_leave_ps": [l0, l1]})

    index = {
        "format": FORMAT,
        "clock_period_ps": int(spec.clock_period_ps),
        "capture_edge_ps": int(spec.capture_edge_ps),
        "pulse_widths_ps": [int(w) for w in spec.pulse_widths_ps],
        "cycles": [int(c) for c in spec.cycles],
        "n_trials": len(spec.cycles) * len(sites) * len(spec.pulse_widths_ps),
        "sites": sites,
        "ffs": ffs,
        "windows": windows,
        "shards": [r["shard"] for r in results],
        "propagation": {"horizon_ps": spec.sim_horizon_ps, "delay_choice": spec.delay_choice,
                        "interconnect": spec.interconnect, "vcs_align": bool(spec.vcs_align)},
        "counts": counts,
        "elapsed_s": round(time.time() - t0, 3),
    }
    tmp = os.path.join(spec.out_dir, "index.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=1)
    os.replace(tmp, os.path.join(spec.out_dir, "index.json"))
    if timing.get("compile") is not None:
        log("[inject] time: " + ", ".join(f"{k} {v:.1f}s" for k, v in timing.items()))
    log(f"[inject] done in {index['elapsed_s']}s: {counts['records']:,} records -> {spec.out_dir}")
    return index


if __name__ == "__main__":  # pragma: no cover
    sys.exit("use the setfi command line")
