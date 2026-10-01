"""Absolute parse-integrity checks on the elaborated netlist.

Comparisons between two runs cannot see a parser defect that damages both runs the same
way (for example, sub-module port bits resolved against the wrong parent bits, which
silently removes nets from the injectable set).  These checks therefore ask questions
that have an answer for ONE netlist:

  A. Does any net name contain unparsed source text?  A net called
     `u_dut/{src1_reg[31:9], n1, src1_reg[7:0]}[8]` is a parser artefact.
  B. Is any net READ by a library cell but driven by nothing -- not an instance
     output, not a primary input, not a constant?  Everything in its forward cone
     would be evaluated from a value the netlist never produces.
  C. Does every sub-module port connection's width agree with the port's declared
     width?  A mismatch means the positional binding applied here cannot be the one
     an elaborator applies.

A and C must be zero; B must not exceed ``BuildSpec.max_undriven_nets`` (default 0).
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Set, Tuple

from .celllib import Cell
from .netlist import ModuleDef, expr_bits, register_net_ranges, resolve_net, whole_conn_bits

# Net names the netlist legitimately contains that are not wires.
_CONSTANT_RE = re.compile(r"^\d+'[sSbBoOhHdD]")


def audit_net_names(net_names: Iterable[str]) -> List[str]:
    """A -- net names that carry unparsed source text."""
    bad = []
    for n in net_names:
        if "{" in n or "}" in n:
            bad.append(n)
        elif n.count("[") != n.count("]"):
            bad.append(n)
    return bad


def driven_and_consumed(modules: Dict[str, ModuleDef], top: str, cells: Dict[str, Cell]
                        ) -> Tuple[Set[str], Set[str], Counter]:
    """(driven, consumed, unreadable cell types) walked over the real hierarchy.

    Deliberately independent of the combinational edge list: that list omits flip-flop
    outputs and black-box outputs, so asking "does anything drive this wire" needs pin
    DIRECTIONS, taken from the library and the netlist's own module declarations.
    """
    lib_out = {name: set(c.outputs) | set(c.inouts) for name, c in cells.items()}
    user_out = {name: {p for p, d in m.port_dirs.items() if d != "input"}
                for name, m in modules.items()}

    driven: Set[str] = set()
    consumed: Set[str] = set()
    unknown: Counter = Counter()
    net_ranges_global: Dict[str, Tuple[int, int]] = {}

    def bits(net: str, net_ranges: Dict[str, Tuple[int, int]]) -> List[str]:
        # net_ranges is the ENCLOSING module's net widths: without it a bare bus inside
        # a concatenation counts as one bit.
        b = expr_bits(net.strip(), net_ranges)
        return b if b is not None else [net.strip()]

    def descend(mname: str, path: str, pin2net: Dict[str, str]) -> None:
        mdef = modules.get(mname)
        if mdef is None:
            return
        register_net_ranges(mdef, path, net_ranges_global)
        for inst in mdef.instances:
            child = f"{path}/{inst.name}" if path else inst.name
            resolved = {p: resolve_net(n, path, mdef.ports, pin2net, mdef.port_ranges,
                                       mdef.net_ranges, net_ranges_global)
                        for p, n in inst.pin2net.items()}
            outs = lib_out.get(inst.celltype, user_out.get(inst.celltype))
            if outs is None:
                unknown[inst.celltype] += 1
                # A cell type that cannot be read is treated as driving EVERY pin it
                # touches: a check must not manufacture orphans out of its own ignorance.
                outs = set(resolved)
            is_hier = inst.celltype in modules and inst.celltype not in lib_out
            for pin, net in resolved.items():
                if pin in outs:
                    driven.update(bits(net, mdef.net_ranges))
                elif not is_hier:
                    consumed.update(bits(net, mdef.net_ranges))
                # A hierarchical instance's INPUT port is not a reader: whether the parent
                # net is consumed is decided inside the child by the leaf pins that read
                # that port.  (Post-layout netlists carry ports that cross-hierarchy
                # optimisation declared and then abandoned; counting the port itself as a
                # reader would refuse a design over a wire nobody evaluates.)
            if is_hier:
                descend(inst.celltype, child, resolved)

    top_def = modules[top]
    descend(top, "", {p: p for p in top_def.ports})
    # The top module's own INPUT ports are driven from outside the design.
    for p, d in top_def.port_dirs.items():
        if d != "input":
            continue
        rng = top_def.port_ranges.get(p)
        if rng is None:
            driven.add(p)
        else:
            msb, lsb = rng
            step = -1 if msb >= lsb else 1
            driven.update(f"{p}[{i}]" for i in range(msb, lsb + step, step))
    return driven, consumed, unknown


def undriven_consumed(driven: Set[str], consumed: Set[str]) -> List[str]:
    """B -- consumed nets with no driver.  Vector/bit granularity: a bare name whose BITS
    are driven is a driven vector (`.op(dec_op)` consumes `dec_op` whole while its driver
    writes `dec_op[2:0]`), and a bit of a wholesale-driven vector is driven too."""
    driven_bases = {n.split("[", 1)[0] for n in driven if "[" in n}
    return sorted(
        n for n in consumed - driven
        if not _CONSTANT_RE.match(n) and n not in ("", "1'b0", "1'b1")
        and n not in driven_bases
        and n.split("[", 1)[0] not in driven)


def audit_port_widths(modules: Dict[str, ModuleDef]) -> List[Dict[str, Any]]:
    """C -- every sub-module port connection's width vs the child port's declared width."""
    bad: List[Dict[str, Any]] = []
    for mname, mdef in modules.items():
        for inst in mdef.instances:
            child = modules.get(inst.celltype)
            if child is None:                     # a library cell -- no body here
                continue
            for pin, net in inst.pin2net.items():
                rng = child.port_ranges.get(pin)
                if rng is None:
                    continue                      # scalar port
                want = abs(rng[0] - rng[1]) + 1
                bits = expr_bits(net.strip(), mdef.net_ranges)
                if bits is None:
                    # Not an expression.  A whole-vector / bit-select / constant
                    # connection still has a definite width.
                    bits = whole_conn_bits(net.strip(), mdef.net_ranges)
                if bits is None:
                    continue                      # genuinely unflattenable
                if len(bits) != want:
                    bad.append({"module": mname, "instance": inst.name,
                                "celltype": inst.celltype, "pin": pin,
                                "declared_bits": want, "connected_bits": len(bits),
                                "connection": net.strip()[:160]})
    return bad


def parse_integrity(net_names: Iterable[str], modules: Dict[str, ModuleDef], top: str,
                    cells: Dict[str, Cell], max_undriven: int,
                    aliases: Dict[str, str] = None) -> Dict[str, Any]:
    """Run A, B and C.  Returns the report; ``report["failures"]`` is empty on pass.
    ``aliases`` maps nets merged by netlist `assign` statements to the surviving net."""
    names = list(net_names)
    bad_names = audit_net_names(names)
    driven, consumed, unknown = driven_and_consumed(modules, top, cells)
    if aliases:
        driven = {aliases.get(n, n) for n in driven}
        consumed = {aliases.get(n, n) for n in consumed}
    orphans = undriven_consumed(driven, consumed)
    width_bad = audit_port_widths(modules)
    fails: List[str] = []
    if bad_names:
        fails.append(f"{len(bad_names)} net name(s) contain unparsed source text "
                     f"(e.g. {bad_names[0][:100]!r}); a netlist construct was not understood")
    if len(orphans) > max_undriven:
        fails.append(f"{len(orphans)} net(s) are read by a cell but driven by nothing "
                     f"(e.g. {orphans[:3]}); set build.max_undriven_nets to accept them")
    if width_bad:
        w = width_bad[0]
        fails.append(f"{len(width_bad)} port connection(s) disagree with the declared "
                     f"port width (e.g. {w['instance']}.{w['pin']}: declared "
                     f"{w['declared_bits']} bits, connected {w['connected_bits']})")
    return {
        "n_nets": len(names),
        "unparsed_net_names": {"n": len(bad_names), "examples": bad_names[:8]},
        "undriven_consumed_nets": {"n": len(orphans), "examples": orphans[:8],
                                   "unreadable_celltypes": dict(unknown)},
        "port_width_mismatches": {"n": len(width_bad), "examples": width_bad[:5]},
        "status": "fail" if fails else "pass",
        "failures": fails,
    }
