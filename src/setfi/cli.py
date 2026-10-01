"""setfi command line.

Stages, in order (each reads the output of the previous ones):

  build     netlist + cell functions + SDF       ->  <dir>/build/
  record    testbench simulation, cycle states   ->  <dir>/record/
  inject    SET campaign, pulse records          ->  <dir>/inject/
  analyze   upset probabilities                  ->  <dir>/analyze/<width model>/

`setfi run` runs every stage that is out of date: a stage is re-run when a
configuration key it reads, one of its input files, or the result of an earlier
stage changed.  <dir>/stages.json records what each stage was run from.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from . import __version__
from . import config as C

STAGES = list(C.STAGES)
STAMPS = "stages.json"
# The file that shows a stage's output is present.
KEY_OUTPUT = {"build": "net_index.json", "record": "cycles.json", "inject": "index.json"}
# What each stage reads from earlier stages: (stage, output digest name).
UPSTREAM = {"build": [], "record": [("build", "nets")],
            "inject": [("build", "graph"), ("record", "states")], "analyze": [("inject", "pulses")]}
INCLUDE_RE = re.compile(r'^\s*`include\s+"([^"]+)"', re.M)


class SetfiCLIError(RuntimeError):
    pass


def paths(cfg) -> Dict[str, str]:
    out = cfg["output"]["dir"]
    return {"out": out, **{st: os.path.join(out, st) for st in STAGES}}


# --------------------------------------------------------------------------- digests
def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _digest_file(path: str) -> str:
    """Content digest; for .npz the arrays, not the archive (which holds timestamps)."""
    if not path.endswith(".npz"):
        return _md5(path)
    import numpy as np
    h = hashlib.md5()
    with np.load(path) as z:
        for k in sorted(z.files):
            a = z[k]
            h.update(f"{k}:{a.dtype.str}:{a.shape}".encode())
            h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _file_content(p: str) -> str:
    return _md5(p) if os.path.isfile(p) else f"<missing {os.path.basename(p)}>"


def _content(sec: str, key: str, value: Any) -> Any:
    """A configuration value as it enters a fingerprint: files by content, not by path,
    so moving a project does not invalidate it."""
    t = C.schema()[sec][key].type
    if value is None:
        return None
    if (sec, key) == ("simulator", "container"):      # an image: identify it by name and size
        return (f"{os.path.basename(value)}:{os.path.getsize(value)}"
                if os.path.isfile(value) else f"<missing {os.path.basename(value)}>")
    if t == "path":
        return _file_content(value)
    if t == "paths":
        return [_file_content(v) for v in value]
    if t == "dirs":                  # the included files count, see _include_files
        return None
    if t == "tables":
        return [{k: (_file_content(v) if k == "rtl_file" else v) for k, v in e.items()}
                for e in value]
    return value


def _include_files(cfg) -> Dict[str, str]:
    """The `include files the testbench reads (recursively), by content."""
    from .pipeline import include_dirs
    dirs = include_dirs(cfg)
    found: Dict[str, str] = {}
    todo = list(cfg["workload"]["testbench"])
    seen = set()
    while todo:
        f = todo.pop()
        if f in seen or not os.path.isfile(f):
            continue
        seen.add(f)
        with open(f, "r", encoding="utf-8", errors="replace") as fh:
            names = INCLUDE_RE.findall(fh.read())
        for n in names:
            for d in [os.path.dirname(f)] + dirs:
                p = os.path.normpath(os.path.join(d, n))
                if os.path.isfile(p):
                    found[n] = _md5(p)
                    todo.append(p)
                    break
    return found


def output_digests(cfg, stage: str) -> Dict[str, str]:
    """Digests of a stage's result, as the later stages read it."""
    d = paths(cfg)[stage]
    if stage == "build":
        from .build import CONTRACT_FILES
        from .pipeline import cell_model_files
        with open(os.path.join(d, "ff_index.json")) as f:
            ffx = json.load(f)
        ff_nets = sorted(set(ffx.get("dnet_to_ffids", {})) | set(ffx.get("qnet_to_ffids", {})))
        # what record reads: the nets, the flip-flop nets, and the cell models it
        # simulates when workload.cell_sim_models is not given
        models = [_file_content(f) for f in cell_model_files(dict(cfg, workload={
            **cfg["workload"], "cell_sim_models": []}), paths(cfg))]
        return {"nets": _sha([_md5(os.path.join(d, "net_index.json")), ff_nets, models]),
                "graph": _sha({n: _digest_file(os.path.join(d, n)) for n in CONTRACT_FILES
                               if os.path.exists(os.path.join(d, n))})}
    if stage == "record":
        from .campaign.state import read_sampled_cycles
        from .pipeline import state_paths
        states, meta = state_paths(d)
        cycles = read_sampled_cycles(meta)
        return {"states": _sha([[c, _md5(os.path.join(states, f"cycle_{c}.hex"))] for c in cycles])}
    if stage == "inject":
        with open(os.path.join(d, "index.json")) as f:
            idx = json.load(f)
        idx.pop("elapsed_s", None)
        for s in idx.get("shards", []):
            s.pop("md5", None)       # an .npz archive holds its write time
        return {"pulses": _sha(idx)}
    return {}


# --------------------------------------------------------------------------- stamps
def _stamps_path(cfg) -> str:
    return os.path.join(paths(cfg)["out"], STAMPS)


def read_stamps(cfg) -> Dict[str, Any]:
    p = _stamps_path(cfg)
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)


def _write_stamps(cfg, stamps: Dict[str, Any]) -> None:
    p = _stamps_path(cfg)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump({st: stamps[st] for st in STAGES if st in stamps}, f, indent=1)
    os.replace(tmp, p)


def _value(cfg, sec: str, key: str) -> Any:
    """A key's value as the stage uses it (a simulator key left out: the preset's)."""
    v = cfg[sec][key]
    if v is None and sec == "simulator":
        from .pipeline import simulator_defaults
        v = simulator_defaults(cfg).get(key)
    return v


def missing_outputs(cfg, stage: str) -> List[str]:
    """Output files of a stage that are not there (only what later stages read)."""
    d = paths(cfg)[stage]
    need: List[str] = []
    try:
        if stage == "build":
            with open(os.path.join(d, "build_manifest.json")) as f:
                need = list(json.load(f)["artifacts"])
        elif stage == "record":
            from .campaign.state import read_sampled_cycles
            need = ["cycles.json"] + [f"states/cycle_{c}.hex"
                                      for c in read_sampled_cycles(os.path.join(d, "cycles.json"))]
        elif stage == "inject":
            with open(os.path.join(d, "index.json")) as f:
                need = ["index.json"] + [s["file"] for s in json.load(f)["shards"]]
    except (OSError, ValueError, KeyError):
        return [KEY_OUTPUT.get(stage, "")]
    return [n for n in need if not os.path.exists(os.path.join(d, n))]


def fingerprint(cfg, stage: str, stamps: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """What a stage's result depends on: the configuration it reads (input files by
    content) and the results of earlier stages."""
    stamps = read_stamps(cfg) if stamps is None else stamps
    conf = {f"{sec}.{key}": _content(sec, key, _value(cfg, sec, key))
            for sec, key in C.stage_keys(stage, cfg)}
    if stage == "record":
        conf["workload.include_files"] = _include_files(cfg)
    upstream = {f"{up}.{name}": (stamps.get(up) or {}).get("outputs", {}).get(name)
                for up, name in UPSTREAM[stage]}
    return {"fingerprint": _sha({"config": conf, "upstream": upstream}), "config": conf,
            "upstream": upstream}


def stage_status(cfg, stage: str, stamps: Dict[str, Any],
                 require_output: bool = True) -> Tuple[bool, str]:
    """(up to date, why not)."""
    s = stamps.get(stage)
    if s is None:
        return False, "it has not run"
    fp = fingerprint(cfg, stage, stamps)
    if s["fingerprint"] != fp["fingerprint"]:
        old = s.get("config", {})
        changed = [k for k in sorted(set(fp["config"]) | set(old))
                   if fp["config"].get(k) != old.get(k)]
        changed += [f"the result of {k.split('.')[0]}" for k in fp["upstream"]
                    if fp["upstream"][k] != s.get("upstream", {}).get(k)]
        return False, "changed: " + ", ".join(changed)
    if require_output:
        if not os.path.isdir(paths(cfg)[stage]):
            return False, f"{paths(cfg)[stage]} is missing"
        miss = missing_outputs(cfg, stage)
        if miss:
            return False, (f"{len(miss)} output file(s) missing, e.g. "
                           f"{os.path.join(paths(cfg)[stage], miss[0])}")
    return True, ""


def _require(cfg, stage: str, for_stage: str) -> None:
    ok, why = stage_status(cfg, stage, read_stamps(cfg))
    if not ok:
        raise SetfiCLIError(f"the '{stage}' stage is not up to date ({why}); run `setfi {stage}` "
                            f"(or `setfi run`) before `setfi {for_stage}`")


def _skip(cfg, stage: str, force: bool) -> bool:
    if not force and stage_status(cfg, stage, read_stamps(cfg))[0]:
        print(f"[{stage}] up to date (--force re-runs it)")
        return True
    return False


def _begin(cfg, stage: str) -> Dict[str, Any]:
    """Drop the stage's stamp before it runs, so a failure cannot leave an old stamp
    that looks current.  Returns the fingerprint it runs from."""
    stamps = read_stamps(cfg)
    fp = fingerprint(cfg, stage, stamps)
    if stamps.pop(stage, None) is not None:
        _write_stamps(cfg, stamps)
    return fp


def _finish(cfg, stage: str, fp: Dict[str, Any], summary: Dict[str, Any]) -> None:
    stamps = read_stamps(cfg)
    stamps[stage] = {"setfi_version": __version__,
                     "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     **fp, "outputs": output_digests(cfg, stage), "summary": summary}
    _write_stamps(cfg, stamps)


# --------------------------------------------------------------------------- stages
def do_build(cfg, force: bool = True) -> None:
    if _skip(cfg, "build", force):
        return
    from .build import run_build
    from .campaign.substrate import load_ff_thresholds
    from .pipeline import build_spec_from_config
    fp = _begin(cfg, "build")
    man = run_build(build_spec_from_config(cfg, paths(cfg)))
    # every observed flip-flop must have a usable sampling window: check it now, not
    # at inject time
    ffs = load_ff_thresholds(paths(cfg)["build"], clock_pins=cfg["library"]["timing_check_clock_pins"],
                             allow_missing=cfg["observe"]["allow_missing_timing_checks"])
    summary = {"instances": man.get("netlist", {}).get("n_instances"),
               "sites": man.get("graph", {}).get("n_sites"), "observed_flip_flops": len(ffs)}
    _finish(cfg, "build", fp, summary)
    print(f"[build] {summary['instances']} instances, {summary['sites']} injection sites, "
          f"{summary['observed_flip_flops']} observed flip-flops -> {paths(cfg)['build']}")


def do_record(cfg, force: bool = True) -> None:
    _require(cfg, "build", "record")
    if _skip(cfg, "record", force):
        return
    from .campaign.state import read_sampled_cycles
    from .pipeline import check_testbench, state_paths, workload_spec_from_config
    from .workload import record_workload
    check_testbench(cfg)
    fp = _begin(cfg, "record")
    record_workload(workload_spec_from_config(cfg, paths(cfg)))
    n = len(read_sampled_cycles(state_paths(paths(cfg)["record"])[1]))
    _finish(cfg, "record", fp, {"cycles": n})
    print(f"[record] {n} cycles recorded -> {paths(cfg)['record']}")


def do_inject(cfg, force: bool = True) -> None:
    _require(cfg, "build", "inject")
    _require(cfg, "record", "inject")
    if _skip(cfg, "inject", force):
        return
    from .campaign.run import check_sites, run_campaign
    from .campaign.substrate import load_ff_thresholds
    from .pipeline import campaign_spec_from_config, clock_period_ps
    check_sites(cfg["fault_model"]["sites"], paths(cfg)["build"])
    ffs = load_ff_thresholds(paths(cfg)["build"], clock_pins=cfg["library"]["timing_check_clock_pins"],
                             allow_missing=cfg["observe"]["allow_missing_timing_checks"])
    setup = max((max(f.setup_rise_ps, f.setup_fall_ps) for f in ffs.values()), default=0)
    if clock_period_ps(cfg) <= setup:
        print(f"[inject] warning: design.clock_period_ns ({clock_period_ps(cfg)} ps) is not longer "
              f"than the largest flip-flop setup time ({setup} ps)")
    try:                      # the records being replaced, if any
        before = output_digests(cfg, "inject")["pulses"]
    except (OSError, ValueError, KeyError):
        before = None
    fp = _begin(cfg, "inject")
    a = paths(cfg)["analyze"]
    try:
        idx = run_campaign(campaign_spec_from_config(cfg, paths(cfg)))
    except BaseException:
        shutil.rmtree(a, ignore_errors=True)     # it described the replaced records
        raise
    _finish(cfg, "inject", fp, idx["counts"])
    if os.path.isdir(a) and read_stamps(cfg)["inject"]["outputs"]["pulses"] != before:
        shutil.rmtree(a)
        print(f"[inject] removed {a}: it described the previous pulse records")


NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def do_analyze(cfg, name: Optional[str] = None, from_option: bool = False) -> List[str]:
    """Analyse the pulse records under every width model of the configuration."""
    from .analysis.report import analyze
    from .analysis.widths import width_model_notes
    stamps = read_stamps(cfg)
    a = cfg["analyze"]
    models = a["width_models"]
    if name is not None:
        if not NAME_RE.match(name):
            raise SetfiCLIError(f"--name {name!r}: use letters, digits, '.', '_' and '-' only")
        if len(models) != 1:
            raise SetfiCLIError("--name needs exactly one width model (give it with --width-model)")
    if not os.path.exists(os.path.join(paths(cfg)["inject"], "index.json")):
        raise SetfiCLIError("there are no pulse records yet; run `setfi inject` (or `setfi run`)")
    miss = missing_outputs(cfg, "inject")
    if miss:
        raise SetfiCLIError(f"{len(miss)} pulse record file(s) are missing, e.g. "
                            f"{os.path.join(paths(cfg)['inject'], miss[0])}; run "
                            f"`setfi inject --force`")
    if "inject" not in stamps:
        print(f"[analyze] note: {STAMPS} has no entry for the campaign in inject/, so it "
              f"cannot be checked against the current inputs")
    else:
        for st in ("build", "record", "inject"):
            ok, why = stage_status(cfg, st, stamps, require_output=False)
            if not ok:
                print(f"[analyze] note: inject/ may not match the current inputs: the {st} "
                      f"stage is out of date ({why}); `setfi run` updates what is needed")
                break
    from .analysis.widths import WidthModelError, width_weights
    from .records import Records
    widths = Records.open(paths(cfg)["inject"]).pulse_widths_ps
    for n, m in enumerate(models):
        try:
            width_weights(m, widths)
        except WidthModelError as e:
            where = "--width-model" if from_option else f"[analyze] width_models, entry {n + 1}"
            raise SetfiCLIError(f"{where}: {e} (the pulse records have the widths {widths})") from None
    outs = []
    for m in models:
        for note in width_model_notes(m, widths):
            print(f"[analyze] note: {note}")
        out = os.path.join(paths(cfg)["analyze"], name or C.analysis_name(m, a["phase_domain"]))
        if os.path.isdir(out):
            try:
                with open(os.path.join(out, "summary.json")) as f:
                    old = json.load(f).get("width_model")
            except (OSError, ValueError):
                old = None
            if old is not None and old != m:
                print(f"[analyze] replacing {out} (it held the width model {json.dumps(old)})")
            shutil.rmtree(out)
        analyze(paths(cfg)["inject"], out, m, phase_domain=a["phase_domain"],
                write_trials=a["write_trials"],
                provenance={"setfi_version": __version__,
                            "pulse_records": output_digests(cfg, "inject")["pulses"],
                            "width_model": m, "phase_domain": a["phase_domain"]})
        print(f"[analyze] -> {out}")
        outs.append(out)
    return outs


def do_status(cfg) -> None:
    """Which stages are up to date, and why the others are not."""
    stamps = read_stamps(cfg)
    pending: List[str] = []           # earlier stages that will run
    for st in STAGES[:-1]:
        ok, why = stage_status(cfg, st, stamps)
        when = (stamps.get(st) or {}).get("finished_utc", "")
        deps = [up for up, _ in UPSTREAM[st] if up in pending]
        if not ok:
            line = f"to run  ({why})"
            pending.append(st)
        elif deps:
            line = f"runs again if the result of {' or '.join(deps)} changes"
            pending.append(st)
        else:
            line = "up to date" + (f"  (last run {when})" if when else "")
        print(f"{st:8s} {line}")
    a = paths(cfg)["analyze"]
    names = sorted(os.listdir(a)) if os.path.isdir(a) else []
    print(f"analyze  {', '.join(names) if names else 'no results'}"
          + ("  (deleted if the pulse records change)" if names and "inject" in pending else ""))


# --------------------------------------------------------------------------- main
def _errors():
    from .analysis.widths import WidthModelError
    from .build import BuildError
    from .campaign.state import StateError
    from .campaign.substrate import SubstrateError
    from .library.liberty import LibertyError
    from .workload.errors import WorkloadError
    return (C.ConfigError, SetfiCLIError, BuildError, WorkloadError, SubstrateError, StateError,
            WidthModelError, LibertyError)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="setfi", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"setfi {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="command")
    p = sub.add_parser("init", help="write a configuration template")
    p.add_argument("path", nargs="?", default="setfi.toml", help="file to write (default setfi.toml)")
    p.add_argument("--force", action="store_true", help="overwrite an existing file")
    helps = {"config": "print the configuration with defaults filled in",
             "status": "show which stages are up to date",
             "build": "build the timing graph (<dir>/build/)",
             "record": "record the circuit state of sampled cycles (<dir>/record/)",
             "inject": "run the SET campaign (<dir>/inject/)",
             "analyze": "compute upset probabilities (<dir>/analyze/)",
             "run": "run every out-of-date stage, then analyze"}
    for cmd, h in helps.items():
        p = sub.add_parser(cmd, help=h, description=h)
        p.add_argument("-c", "--config", required=True, metavar="FILE", help="configuration file")
        if cmd == "analyze":
            p.add_argument("--width-model", metavar="JSON",
                           help='analyse this width model instead of [analyze] width_models, e.g. '
                                '\'{"type": "exponential", "tau_ps": 40}\'')
            p.add_argument("--name", help="directory name under <dir>/analyze/ (default: from "
                                          "the width model)")
        if cmd in ("build", "record", "inject", "run"):
            p.add_argument("--force", action="store_true",
                           help="re-run even if up to date" if cmd != "run" else "re-run every stage")
    args = ap.parse_args(argv)

    if args.cmd == "init":
        if os.path.exists(args.path) and not args.force:
            print(f"setfi: {args.path} exists (use --force to overwrite)", file=sys.stderr)
            return 1
        with open(args.path, "w") as f:
            f.write(C.template())
        print(f"wrote {args.path}")
        return 0

    try:
        sys.stdout.reconfigure(line_buffering=True)     # progress and errors in order
    except (AttributeError, ValueError):
        pass
    try:
        overrides: Dict[str, Dict[str, Any]] = {}
        if getattr(args, "width_model", None):
            try:
                overrides = {"analyze": {"width_models": [json.loads(args.width_model)]}}
            except json.JSONDecodeError as e:
                raise C.ConfigError(f"--width-model is not valid JSON: {e}") from None
        if not os.path.exists(args.config):
            raise C.ConfigError(f"configuration file {args.config} not found "
                                f"(`setfi init` writes a template)")
        cfg = C.load(args.config, overrides=overrides,
                     check_files=args.cmd not in ("analyze", "config", "status"),
                     check_width_models=args.cmd != "analyze")
        if args.cmd == "config":
            print(C.dump_resolved(cfg))
            missing = C.missing_files(cfg)
            for m in missing:
                print(f"# not found: {m}")
            return 1 if missing else 0
        elif args.cmd == "status":
            do_status(cfg)
        elif args.cmd == "build":
            do_build(cfg, args.force)
        elif args.cmd == "record":
            do_record(cfg, args.force)
        elif args.cmd == "inject":
            do_inject(cfg, args.force)
        elif args.cmd == "analyze":
            do_analyze(cfg, args.name, from_option=bool(args.width_model))
        elif args.cmd == "run":
            do_build(cfg, args.force)
            do_record(cfg, args.force)
            do_inject(cfg, args.force)
            do_analyze(cfg)
    except _errors() as e:
        print(f"setfi {args.cmd}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
