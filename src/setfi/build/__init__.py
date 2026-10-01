"""Build stage: gate-level netlist + SDF + behavioural cell library -> substrate.

    from setfi.build import BuildSpec, LibrarySpec, run_build

    spec = BuildSpec(netlist="top.mapped.v", sdf="top.mapped.sdf", top="top",
                     out_dir="runs/top/substrate",
                     library=LibrarySpec(behavioral_verilog="cells.behavioral.v"))
    manifest = run_build(spec)

The output directory holds the files the campaign reads (``CONTRACT_FILES``) and
``build_manifest.json`` (input checksums, options, counts, check results).

Modules:
    spec      BuildSpec / LibrarySpec / BuildError
    netlist   structural-Verilog parser, hierarchical elaboration, SDF cross-check
    celllib   behavioural library: pins, sequential inference, cell functions
    graph     edges, FF index, injection sites, per-site cones
    timing    SDF -> delay table, arc index, cell arcs, timing checks, interconnect
    observe   which FFs are observation points
    checks    absolute parse-integrity checks
    pipeline  run_build
"""
from .pipeline import CONTRACT_FILES, MANIFEST_NAME, run_build
from .spec import BuildError, BuildSpec, LibrarySpec

__all__ = ["BuildSpec", "LibrarySpec", "BuildError", "run_build", "CONTRACT_FILES",
           "MANIFEST_NAME"]
