"""``python -m setfi.workload <workload.json>``: record one workload.

The JSON object holds :class:`WorkloadSpec` fields; ``simulator`` is either
``{"preset": "vcs" | "icarus", <preset arguments>..., "overrides": {...}}`` or
explicit :class:`SimulatorSpec` fields.
"""
from __future__ import annotations

import argparse
import json
import sys

from .errors import WorkloadError
from .record import record_workload
from .spec import WorkloadSpec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m setfi.workload", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="workload JSON")
    ap.add_argument("--output-dir", help="override output_dir")
    args = ap.parse_args(argv)
    with open(args.config) as f:
        d = json.load(f)
    if args.output_dir:
        d["output_dir"] = args.output_dir
    try:
        res = record_workload(WorkloadSpec.from_dict(d), log=lambda m: print(m, flush=True))
    except WorkloadError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1
    print(json.dumps(res["outputs"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
