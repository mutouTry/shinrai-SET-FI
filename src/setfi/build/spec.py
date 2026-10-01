"""User-facing configuration of the build stage.

Two dataclasses, one per config section:

``LibrarySpec``  (``[library]``) -- everything that is a property of the standard-cell
                 library / PDK / memory compiler: the behavioural cell file, the pin-name
                 conventions used to recognise flip-flops, which cells carry no timing arc,
                 which cells are opaque macros, which sequential cells are clock gates.
``BuildSpec``    (``[build]``)   -- the design inputs, the output directory, and the
                 policies of the build (injection sites, SDF rail, side pins, gates).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, fields
from typing import Any, Dict, Mapping, Tuple


class BuildError(RuntimeError):
    """The inputs cannot be turned into a trustworthy timing graph.

    Raised instead of guessing: each of these conditions would otherwise give a silently
    wrong graph."""


def _tuple_of_str(name: str, v: Any) -> Tuple[str, ...]:
    if isinstance(v, str):
        raise BuildError(f"{name}: expected a list of strings, got the string {v!r}")
    try:
        out = tuple(v)
    except TypeError:
        raise BuildError(f"{name}: expected a list of strings, got {v!r}") from None
    for x in out:
        if not isinstance(x, str):
            raise BuildError(f"{name}: expected a list of strings, got element {x!r}")
    return out


def _check_regexes(name: str, pats: Tuple[str, ...]) -> None:
    for p in pats:
        try:
            re.compile(p)
        except re.error as exc:
            raise BuildError(f"{name}: invalid regular expression {p!r}: {exc}") from None


@dataclass
class LibrarySpec:
    """Conventions of the cell library.  Section ``[library]``."""

    # The behavioural standard-cell library: an assign-style Verilog file in which every
    # combinational cell is written with `assign` and every state-holding cell with
    # `always` (NOT gate primitives / UDPs -- a timing-model library parses, but yields no
    # cell functions).  It is the only source of cell functions.
    behavioral_verilog: str

    # ---- sequential-cell inference (a cell is sequential iff it has a Q-like output, a
    # D-like input and a clock pin) ----
    # Output pins taken as the state outputs, in preference order.  If none is present,
    # every output is taken.
    q_pin_names: Tuple[str, ...] = ("Q", "QN", "QB", "Q_N", "QBAR")
    # Data input, first match wins.  If none is present, the alphabetically first input
    # that is neither the clock nor an async pin is used.
    d_pin_names: Tuple[str, ...] = ("D", "DI", "DIN", "DATA", "I")
    # Clock pin names, first match wins.  Consulted only after the cell's own
    # `always @(posedge X)` and specify-block arcs (which name the clock directly).
    # NOTE: this is also what classifies a cell as sequential at all; a stateful cell
    # whose clock/enable pin is not reachable by any of the three routes (e.g. `CPN`,
    # `EN`) is caught by the build's cell-function check and must be added here.
    clock_pin_names: Tuple[str, ...] = ("CP", "CK", "CLK", "CLKN", "G", "E")
    # Asynchronous set/reset pins, never taken as the data pin (compared upper-case).
    async_pin_names: Tuple[str, ...] = ("RN", "SN", "R", "S", "RESET", "SET",
                                        "CDN", "SDN", "CLRN", "PRN")
    # A sequential arc from one of these pins is labelled `seq_scan_enable` in cell arcs.
    scan_enable_pin_names: Tuple[str, ...] = ("SCE",)

    # ---- SDF timing checks ----
    # For a scan FF with a check on one data pin but not on its twin (D vs scan-data),
    # copy the check across.  Each pair is (data pin, scan-data pin).
    mirror_scan_data_checks: bool = True
    scan_data_pin_pairs: Tuple[Tuple[str, str], ...] = (("D", "SCD"),)

    # ---- netlist <-> SDF cross-validation ----
    # Cells that carry no timing arc and are therefore absent from the SDF.  A cell with
    # no input pin or no output pin (tie cells, bus holders, fillers, decaps, antenna
    # diodes) is recognised as untimed structurally; list further cell names here
    # (regular expressions, full match).
    untimed_cells: Tuple[str, ...] = ()
    # Opaque macros (memories, hard IP): instantiated in the netlist, defined neither in
    # the behavioural library nor as a netlist module.  Their outputs seed the forward
    # region but are never injection sites.  An instance of a cell type that is in none
    # of the three places is an error.  Regular expressions, full match.
    blackbox_cells: Tuple[str, ...] = ()

    # ---- observed flip-flops ----
    # Sequential cell types whose capture is not an observation point (integrated clock
    # gates look like FFs to the inference above).  Regular expressions matched against the
    # whole cell name; prefix `(?i)` for case-insensitive.
    non_observed_cells: Tuple[str, ...] = ()
    # An observed FF must capture through one of these data pins.
    observed_d_pins: Tuple[str, ...] = ("D",)

    def validate(self) -> None:
        if not isinstance(self.behavioral_verilog, str) or not self.behavioral_verilog:
            raise BuildError("library.cell_functions is required")
        if not os.path.isfile(self.behavioral_verilog):
            raise BuildError(f"library.cell_functions: not a file: {self.behavioral_verilog}")
        for f in ("q_pin_names", "d_pin_names", "clock_pin_names", "async_pin_names",
                  "scan_enable_pin_names", "untimed_cells", "blackbox_cells",
                  "non_observed_cells", "observed_d_pins"):
            setattr(self, f, _tuple_of_str(f"library.{f}", getattr(self, f)))
        pairs = []
        for p in self.scan_data_pin_pairs:
            p = tuple(p)
            if len(p) != 2 or not all(isinstance(x, str) and x for x in p):
                raise BuildError(f"library.scan_data_pin_pairs: each entry must be two pin "
                                 f"names, got {p!r}")
            pairs.append(p)
        self.scan_data_pin_pairs = tuple(pairs)
        for f in ("untimed_cells", "blackbox_cells", "non_observed_cells"):
            _check_regexes(f"library.{f}", getattr(self, f))
        if not self.observed_d_pins:
            raise BuildError("observe.data_pins must name at least one pin")


@dataclass
class BuildSpec:
    """Inputs and policies of one build.  Section ``[build]`` (+ ``library``)."""

    netlist: str            # gate-level netlist (structural Verilog)
    sdf: str                # SDF written for exactly this netlist (DESIGN must equal `top`)
    top: str                # top module name
    out_dir: str            # output directory (created)
    library: LibrarySpec

    # ---- injection sites ----
    # Drop FF Q nets from the sites (a Q net that is also another FF's D net is kept:
    # a direct FF1.Q -> FF2.D wire is a legitimate D injection point).
    exclude_q_from_src: bool = True
    # Drop FF D nets from the sites.  Default keeps them: an SET on the wire into D is
    # physical and yields a single-FF pattern no upstream site produces.
    exclude_d_from_src: bool = False

    # ---- SDF consumption ----
    # Which end of every (min:typ:max) triplet is used: "late" (max) or "early" (min).
    sdf_rail: str = "late"
    # Every delay is rounded (half up) to a multiple of this many picoseconds.
    time_precision_ps: int = 1

    # ---- side pins of conditional (COND) delay arcs ----
    side_pins_topk: int = 10            # at most this many side pins per arc
    exclude_src_from_side: bool = True  # the arc's own input is never a side pin
    max_enum_side_pins: int = 10        # arcs with more side pins get no semantics

    # ---- observed flip-flops (see also library.non_observed_cells/observed_d_pins) ----
    # FF instances whose name contains one of these substrings are not observed
    # (synthesis-inserted clock-gating registers).
    non_observed_instances: Tuple[str, ...] = ()

    # ---- self-check: rebuild this many sampled cones by plain BFS and compare ----
    self_check: bool = True
    self_check_samples: int = 50
    self_check_seed: int = 1

    # ---- gates ----
    # Netlist leaves vs SDF instances must match exactly (else BuildError).
    strict_sdf_crosscheck: bool = True
    # Parse-integrity check B: nets read by a cell but driven by nothing, allowed count.
    max_undriven_nets: int = 0

    # Also write the intermediate products (cone database, super-edges, arc delays table, ...)
    # under <out_dir>/intermediate/.
    keep_intermediates: bool = False
    # Accept gate arcs inside injection cones that have no SDF delay (pulses stop there).
    allow_missing_delays: bool = False

    def validate(self) -> None:
        if not isinstance(self.library, LibrarySpec):
            raise BuildError("BuildSpec.library must be a LibrarySpec")
        self.library.validate()
        for f in ("netlist", "sdf"):
            v = getattr(self, f)
            if not isinstance(v, str) or not v:
                raise BuildError(f"design.{f} is required")
            if not os.path.isfile(v):
                raise BuildError(f"design.{f}: not a file: {v}")
        if not isinstance(self.top, str) or not re.fullmatch(r"[A-Za-z_]\w*", self.top or ""):
            raise BuildError(f"design.top: not a module name: {self.top!r}")
        if not isinstance(self.out_dir, str) or not self.out_dir:
            raise BuildError("build.out_dir is required")
        rail = str(self.sdf_rail).strip().lower()
        if rail not in ("late", "early"):
            raise BuildError(f"simulation.sdf_rail must be 'late' or 'early', got {self.sdf_rail!r}")
        self.sdf_rail = rail
        for f, lo in (("time_precision_ps", 1), ("side_pins_topk", 0),
                      ("max_enum_side_pins", 0), ("self_check_samples", 0),
                      ("max_undriven_nets", 0)):
            v = getattr(self, f)
            if isinstance(v, bool) or not isinstance(v, int) or v < lo:
                raise BuildError(f"build.{f} must be an integer >= {lo}, got {v!r}")
        self.non_observed_instances = _tuple_of_str("build.non_observed_instances",
                                                    self.non_observed_instances)

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, build: Mapping[str, Any],
                  library: Mapping[str, Any] | None = None) -> "BuildSpec":
        """Build a spec from config mappings (e.g. the ``[build]`` and ``[library]``
        tables of a TOML file).  ``library`` may also be given as ``build["library"]``.
        Unknown keys are an error, not ignored."""
        b = dict(build)
        lib = dict(library if library is not None else b.pop("library", {}) or {})
        b.pop("library", None)
        lib_names = {f.name for f in fields(LibrarySpec)}
        unknown = sorted(set(lib) - lib_names)
        if unknown:
            raise BuildError(f"unknown [library] key(s): {unknown}")
        build_names = {f.name for f in fields(cls)} - {"library"}
        unknown = sorted(set(b) - build_names)
        if unknown:
            raise BuildError(f"unknown [build] key(s): {unknown}")
        if "scan_data_pin_pairs" in lib:
            lib["scan_data_pin_pairs"] = tuple(tuple(p) for p in lib["scan_data_pin_pairs"])
        for k, v in list(lib.items()):
            if isinstance(v, list):
                lib[k] = tuple(v)
        for k, v in list(b.items()):
            if isinstance(v, list):
                b[k] = tuple(v)
        try:
            libspec = LibrarySpec(**lib)
            return cls(library=libspec, **b)
        except TypeError as exc:   # a required key is missing
            raise BuildError(str(exc)) from None

    def to_dict(self) -> Dict[str, Any]:
        """Plain-data view (for the manifest)."""
        out: Dict[str, Any] = {}
        for f in fields(self):
            if f.name == "library":
                continue
            v = getattr(self, f.name)
            out[f.name] = list(v) if isinstance(v, tuple) else v
        lib: Dict[str, Any] = {}
        for f in fields(self.library):
            v = getattr(self.library, f.name)
            if f.name == "scan_data_pin_pairs":
                v = [list(p) for p in v]
            elif isinstance(v, tuple):
                v = list(v)
            lib[f.name] = v
        out["library"] = lib
        return out
