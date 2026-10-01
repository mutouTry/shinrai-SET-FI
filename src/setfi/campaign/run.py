"""The injection campaign: every (sampled cycle, site, pulse width) trial.

For each trial the SET is injected on the site's net at t = 0 of a simulation
that starts from the recorded state of that cycle, propagated through the
site's fan-out cone, and every pulse that reaches the D pin of an observed FF
is written as a record (see ``setfi.records``).  The SET start time within the
cycle is not simulated separately: it only shifts the pulse, and the capture
condition is applied analytically by the analysis step.

Optionally the campaign also writes the classic discrete view: for a grid of
start times, which FFs capture (``phase_grid_captures.csv``).
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

import numpy as np

from .. import _engine
from ..records import FORMAT, INF_PS, write_shard
from .cone import (ConeStats, CapturePlan, PackedCone, build_capture_plan, build_cone,
                   pack_cone, thresholds)
from .state import read_cycle_bits
from .substrate import Substrate, load_substrate

STAT_KEYS = ("applied_events", "canceled_late", "canceled_race", "canceled_replace",
             "canceled_back_to_current", "kept_same_target", "coalesced_nets", "coalesced_events",
             "dropped_arc_no_delay", "dropped_no_candidate_arc")


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
    delay_choice: str = "max"                 # "max" | "min" over inputs that explain a change
    electrical_masking: bool = True
    em_margin_ps: int = 0
    coalesce_window_ps: int = 0
    interconnect: str = "ignore"              # "ignore" | "transport"
    clock_pins: List[str] = field(default_factory=lambda: ["CP", "CK", "CLK"])
    allow_missing_timing_checks: bool = False
    sites: Optional[List[str]] = None         # subset of injection sites (default: all)
    cycles_per_shard: int = 10
    jobs: int = 1
    phase_grid_steps: Optional[int] = None    # write phase_grid_captures.csv on k*T/steps
    timeprecision_ps: int = 1

    def resolved(self) -> "CampaignSpec":
        s = CampaignSpec(**asdict(self))
        if s.capture_edge_ps is None:
            s.capture_edge_ps = int(s.clock_period_ps)
        if s.sim_horizon_ps is None:
            s.sim_horizon_ps = int(Fraction(5, 4) * s.capture_edge_ps)
        return s


def phase_grid(period_ps: int, steps: int) -> np.ndarray:
    """k * T / steps for k = 0..steps, rounded half up to integer ps."""
    out = []
    for k in range(steps + 1):
        v = Fraction(k * period_ps, steps)
        out.append(int(v.numerator * 2 + v.denominator) // (2 * v.denominator))
    return np.asarray(out, dtype=np.int64)


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
    """One process: a fixed slice of the injection sites, all cycles."""

    def __init__(self, spec: CampaignSpec, sub: Substrate, sites: List[str], site_index: Dict[str, int]):
        self.spec = spec
        self.sub = sub
        self.sites = sites
        self.site_idx = site_index
        self.cone_stats = ConeStats()
        self.cones: Dict[str, CompiledSite] = {}
        self.pws = np.asarray(spec.pulse_widths_ps, dtype=np.int64)
        self.starts = (phase_grid(spec.capture_edge_ps, spec.phase_grid_steps)
                       if spec.phase_grid_steps else None)

    def cone(self, site: str) -> CompiledSite:
        c = self.cones.get(site)
        if c is None:
            sub = self.sub
            cone = build_cone(sub, site, self.cone_stats)
            plan = build_capture_plan(sub, cone, self.spec.capture_edge_ps, self.cone_stats)
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
        grid_lines: List[str] = []
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
                    bool(spec.electrical_masking), int(spec.em_margin_ps),
                    spec.delay_choice == "max")
                sidx = self.site_idx[site]
                per_start: Optional[List[List[str]]] = (
                    [[] for _ in range(len(self.starts))] if self.starts is not None else None)
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
                    if per_start is not None:
                        mw = _engine.capture_masks(
                            self.starts, plan.thr_enter, plan.thr_leave, ev_ptr, ev_t, ev_v,
                            plan.dp_rec_slot, dp_base, plan.dp_ptr, plan.dp_idx, plan.word_index,
                            plan.word_mask, plan.n_words, INF_PS)
                        nz = np.nonzero(np.any(mw != 0, axis=1))[0]
                        for si in nz.tolist():
                            ids = _decode(mw[si], plan.ffids)
                            per_start[si].append(f"{cyc},{site},{int(self.starts[si])},{pw},{ids}\n")
                if per_start is not None:
                    for lines in per_start:
                        grid_lines.extend(lines)

        cols = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in rec.items()}
        shard = write_shard(os.path.join(spec.out_dir, f"records_{shard_id}.npz"), cols)
        shard.update(cycles=[int(c) for c in cycles])
        if self.starts is not None:
            with open(os.path.join(spec.out_dir, f"grid_{shard_id}.csv.part"), "w") as f:
                f.write("".join(grid_lines))
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


def _decode(words: np.ndarray, ffids: np.ndarray) -> str:
    out = []
    n = int(ffids.shape[0])
    for wi in range(int(words.shape[0])):
        w = int(words[wi])
        while w:
            lsb = w & -w
            b = wi * 64 + lsb.bit_length() - 1
            if b < n:
                out.append(str(int(ffids[b])))
            w ^= lsb
    return "|".join(out)


_G: Dict[str, object] = {}


def _run_slice_in_child(chunk: int):
    spec, sub, slices, site_index, blocks = (_G[k] for k in ("spec", "sub", "slices", "site_index", "blocks"))
    w = _Worker(spec, sub, slices[chunk], site_index)
    return w.run_all(blocks, chunk, len(slices))


def check_sites(sites: List[str], stage_dir: str) -> None:
    """Refuse site names that are not injection sites, before anything is deleted."""
    if not sites:
        return
    import difflib
    from .substrate import SubstrateError, load_net_index
    net_to_idx, idx_to_net = load_net_index(os.path.join(stage_dir, "net_index.json"))
    site_idx = set()
    with open(os.path.join(stage_dir, "site_cones.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                site_idx.add(int(json.loads(line)["src_net_idx"]))
    names = {idx_to_net[i] for i in site_idx}
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
    t0 = time.time()
    sub = load_substrate(spec.stage_dir, clock_pins=spec.clock_pins, interconnect=spec.interconnect,
                         timeprecision_ps=spec.timeprecision_ps,
                         allow_missing_timing_checks=spec.allow_missing_timing_checks)
    all_sites = sorted(sub.skeleton)
    if spec.sites:
        check_sites(list(spec.sites), spec.stage_dir)
        sites = sorted(set(spec.sites))
    else:
        sites = all_sites
    site_index = {s: i for i, s in enumerate(sites)}
    os.makedirs(spec.out_dir, exist_ok=True)
    for f in os.listdir(spec.out_dir):          # a previous campaign's files must not survive
        if f.startswith(("records_", "grid_")) or f in ("index.json", "phase_grid_captures.csv"):
            os.remove(os.path.join(spec.out_dir, f))
    log(f"[inject] {len(spec.cycles)} cycles x {len(sites)} sites x {len(spec.pulse_widths_ps)} widths "
        f"= {len(spec.cycles) * len(sites) * len(spec.pulse_widths_ps):,} trials; "
        f"{sub.counts['n_ff_observed']} observed FFs; T = {spec.clock_period_ps} ps")

    blocks = [(b, spec.cycles[i:i + spec.cycles_per_shard])
              for b, i in enumerate(range(0, len(spec.cycles), spec.cycles_per_shard))]
    # Each process owns a fixed, contiguous slice of the sites for every cycle, so it
    # builds (and holds) only the cones of its own sites.
    n_chunks = max(1, min(int(spec.jobs), len(sites)))
    bounds = [round(i * len(sites) / n_chunks) for i in range(n_chunks + 1)]
    slices = [sites[bounds[i]:bounds[i + 1]] for i in range(n_chunks)]
    results: List[Dict] = []
    if n_chunks > 1:
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

    if spec.phase_grid_steps:
        grid_path = os.path.join(spec.out_dir, "phase_grid_captures.csv")
        with open(grid_path, "w") as out:
            out.write("cycle,site,start_ps,width_ps,ffs\n")
            for r in results:
                part = os.path.join(spec.out_dir, "grid_" + r["shard"]["file"][len("records_"):-len(".npz")]
                                    + ".csv.part")
                with open(part) as f:
                    for chunk in iter(lambda: f.read(1 << 22), ""):
                        out.write(chunk)
                os.remove(part)

    ffs = []
    for ffid in sorted(sub.ffs):
        ff = sub.ffs[ffid]
        e0, l0, e1, l1 = thresholds(sub, ffid, spec.capture_edge_ps)
        ffs.append({"id": ffid, "inst": ff.inst, "d_net": ff.d_net, "d_pin": ff.d_pin,
                    "setup_rise_ps": ff.setup_rise_ps, "hold_rise_ps": ff.hold_rise_ps,
                    "setup_fall_ps": ff.setup_fall_ps, "hold_fall_ps": ff.hold_fall_ps,
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
        "shards": [r["shard"] for r in results],
        "propagation": {"horizon_ps": spec.sim_horizon_ps, "delay_choice": spec.delay_choice,
                        "electrical_masking": spec.electrical_masking,
                        "em_margin_ps": spec.em_margin_ps, "interconnect": spec.interconnect},
        "counts": counts,
        "elapsed_s": round(time.time() - t0, 3),
    }
    tmp = os.path.join(spec.out_dir, "index.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=1)
    os.replace(tmp, os.path.join(spec.out_dir, "index.json"))
    log(f"[inject] done in {index['elapsed_s']}s: {counts['records']:,} records -> {spec.out_dir}")
    return index


if __name__ == "__main__":  # pragma: no cover
    sys.exit("use the setfi command line")
