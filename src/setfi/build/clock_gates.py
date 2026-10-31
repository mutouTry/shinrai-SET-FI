"""The integrated clock gates of a netlist and what they clock (clock_gates.json).

Written with ``observe.clock_gates = "expand"``.  A gate is an instance of a cell
that Liberty declares with ``clock_gating_integrated_cell``.  The gates of the types
``latch_posedge*`` are modelled: the latch is transparent while the clock is low, so
the value of the enable at the rising edge decides whether the gate passes that edge.

From the gate's output the clock is followed through combinational cells up to the
next sequential cells and black boxes.  What this finds:

    ffs            flip-flops that sample the rising edge the gate passes
    gates          modelled gates in series behind this one
    blackbox_pins  pins of black boxes in the clock fan-out
    other_edge     sequential cells that do not sample that rising edge (level-sensitive
                   cells, the falling edge, an unclear polarity)

A cell behind several modelled gates belongs to none of them (``several_gates``).
Other sequential cells that drive a clock pin, and gates of other types, are not
modelled: ``other_clock_sources`` and ``unmodelled_gates`` count them.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

from .celllib import Cell, CellFunctions, sequential_traits
from .spec import BuildError

FORMAT = "setfi-clock-gates/1"
MODELLED_TYPE = re.compile(r"latch_posedge(_precontrol|_postcontrol)?(_obs)?")
TIED_0, TIED_1, STORED_VALUE = -1, -2, -3


def gate_instances(insts: Dict[str, Any], cells: Dict[str, Cell],
                   gate_cells: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """{instance: pin roles} of the gates of the netlist, checked against how the build
    reads the cell: enable as its data pin, gate clock as its clock pin."""
    out: Dict[str, Dict[str, Any]] = {}
    checked: Set[str] = set()
    for path, inst in insts.items():
        roles = gate_cells.get(inst.celltype)
        if roles is None:
            continue
        if inst.celltype not in checked:
            checked.add(inst.celltype)
            c = cells[inst.celltype]
            if not c.is_sequential or c.clock_pin != roles["clock"] or c.d_pins != (roles["enable"],):
                raise BuildError(
                    f"clock gate {inst.celltype}: Liberty names clock pin {roles['clock']} and enable "
                    f"pin {roles['enable']}, but the build reads clock pin {c.clock_pin} and data pins "
                    f"{list(c.d_pins)}. Set library.clock_pins and library.d_pins so that they agree.")
        out[path] = roles
    return out


def _clock_fanout(net: str, readers: Dict[str, List[Tuple[str, str]]], insts: Dict[str, Any],
                  cells: Dict[str, Cell], funcs: CellFunctions, blackboxes: Set[str]):
    """Follow a clock net through combinational cells.  Returns ({sequential instance:
    set of (inverted or None if unclear, through a cell with other inputs)}, {black-box pin})."""
    seq: Dict[str, Set[Tuple[Any, bool]]] = defaultdict(set)
    bb_pins: Set[str] = set()
    seen = set()
    todo = [(net, False, False)]
    while todo:
        state = todo.pop()
        if state in seen:
            continue
        seen.add(state)
        n, inverted, logic = state
        for path, pin in readers.get(n, ()):
            if path in blackboxes:
                bb_pins.add(f"{path}.{pin}")
                continue
            inst = insts[path]
            cell = cells[inst.celltype]
            if cell.is_sequential:
                if pin == cell.clock_pin:
                    seq[path].add((inverted, logic))
                continue
            several = sum(p in inst.pin2net for p in cell.inputs) > 1
            for out in cell.outputs:
                if out not in inst.pin2net:
                    continue
                sem = funcs.arc(inst.celltype, pin, out)
                if sem is None:
                    continue
                rise = bool(sem.event_patterns["posedge"]["rise"])
                fall = bool(sem.event_patterns["posedge"]["fall"])
                inv = None if inverted is None or rise == fall else (inverted ^ fall)
                todo.append((inst.pin2net[out], inv, logic or several))
    return seq, bb_pins


def _next_net(pin: str, pin2net: Dict[str, str], cell: Cell, net_to_idx: Dict[str, int]) -> int:
    """A pin of the next-value table as clock_gates.json holds it: a net index, or
    STORED_VALUE, TIED_0 or TIED_1."""
    if pin not in cell.inputs:
        return STORED_VALUE
    net = pin2net.get(pin, "")
    if net in net_to_idx:
        return net_to_idx[net]
    return TIED_1 if net.endswith("'b1") else TIED_0


def analyze_clock_gates(insts: Dict[str, Any], cells: Dict[str, Cell], funcs: CellFunctions,
                        blackboxes: List[str], gate_cells: Dict[str, Dict[str, Any]],
                        ff_index: Dict[str, Any], net_to_idx: Dict[str, int]) -> Dict[str, Any]:
    """The content of clock_gates.json.  ``ff_index``: before the observation filter."""
    bbs = set(blackboxes)
    ffid_of = {e["ff_inst"]: int(i) for i, e in ff_index["ffid_to_info"].items()}
    all_gates = gate_instances(insts, cells, gate_cells)
    gates = {p: r for p, r in all_gates.items() if MODELLED_TYPE.fullmatch(str(r["type"])) and p in ffid_of}
    unmodelled = {p: r for p, r in all_gates.items() if p not in gates}

    readers: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    driver: Dict[str, str] = {}
    for path, inst in insts.items():
        cell = cells.get(inst.celltype)
        if cell is None:
            if path in bbs:
                for pin, net in inst.pin2net.items():
                    readers[net].append((path, pin))
            continue
        for pin in cell.inputs:
            if pin in inst.pin2net:
                readers[inst.pin2net[pin]].append((path, pin))
        for pin in cell.outputs:
            if pin in inst.pin2net:
                driver[inst.pin2net[pin]] = path

    fanout = {g: _clock_fanout(insts[g].pin2net[r["out"]], readers, insts, cells, funcs, bbs)
              if r["out"] in insts[g].pin2net else ({}, set()) for g, r in gates.items()}
    parents: Dict[str, Set[str]] = defaultdict(set)
    for g, (seq, _) in fanout.items():
        for path in seq:
            parents[path].add(g)
    several = sorted(p for p, gs in parents.items() if len(gs) > 1)

    rows: Dict[str, Dict[str, Any]] = {}
    ffs: Dict[str, Dict[str, Any]] = {}
    next_tables: Dict[str, List[int]] = {}
    for g, (seq, bb_pins) in fanout.items():
        inst = insts[g]
        roles = gates[g]
        test = inst.pin2net.get(roles["test"]) if roles["test"] else None
        row: Dict[str, Any] = {
            "inst": g, "celltype": inst.celltype, "type": roles["type"], "ffid": ffid_of[g],
            "enable_net_idx": net_to_idx[inst.pin2net[roles["enable"]]],
            "test_net_idx": net_to_idx.get(test) if test else None,
            "parent": None, "gates": [], "ffs": [], "blackbox_pins": sorted(bb_pins), "other_edge": [],
            "through_logic": 0}
        rows[g] = row
        for path in sorted(seq):
            if len(parents[path]) > 1:
                continue
            polarity = {inverted for inverted, _ in seq[path]}
            inverted = polarity.pop() if len(polarity) == 1 else None
            row["through_logic"] += any(logic for _, logic in seq[path])
            if path in gates:
                row["gates" if inverted is False else "other_edge"].append(path)
                continue
            traits = sequential_traits(cells[insts[path].celltype])
            rising = (inverted is not None and traits.edge is not None
                      and (traits.edge == "posedge") != inverted)
            if not rising or traits.q_pin not in insts[path].pin2net or path not in ffid_of:
                row["other_edge"].append(path)
                continue
            pin2net = insts[path].pin2net
            info = ff_index["ffid_to_info"][str(ffid_of[path])]
            row["ffs"].append(ffid_of[path])
            ffs[str(ffid_of[path])] = {
                "inst": path, "celltype": insts[path].celltype, "gate": g,
                "d_net_idx": net_to_idx[info["d_net"]],
                "q_net_idx": net_to_idx[pin2net[traits.q_pin]], "q_inverted": traits.q_inverted,
                "async": [[net_to_idx[pin2net[p]], level] for p, level in traits.async_pins
                          if pin2net.get(p) in net_to_idx]}
            if traits.next_table is not None:
                ffs[str(ffid_of[path])]["next_nets"] = [
                    _next_net(p, pin2net, cells[insts[path].celltype], net_to_idx) for p in traits.next_pins]
                next_tables[insts[path].celltype] = list(traits.next_table)
    for g, row in rows.items():
        for child in row["gates"]:
            rows[child]["parent"] = g

    # parents before children: the expansion computes enables down the chains
    order: List[str] = []
    todo = sorted((g for g, r in rows.items() if r["parent"] is None), reverse=True)
    while todo:
        g = todo.pop()
        order.append(g)
        todo.extend(reversed(rows[g]["gates"]))

    other: Dict[str, Dict[str, Any]] = {}
    memo: Dict[str, Set[str]] = {}

    def sources(net: str) -> Set[str]:
        got = memo.get(net)
        if got is not None:
            return got
        found: Set[str] = set()
        seen = set()
        todo = [net]
        while todo:
            n = todo.pop()
            if n in seen:
                continue
            seen.add(n)
            d = driver.get(n)
            if d is None:
                continue
            cell = cells[insts[d].celltype]
            if cell.is_sequential:
                found.add(d)
            else:
                todo.extend(insts[d].pin2net[p] for p in cell.inputs if p in insts[d].pin2net)
        memo[net] = found
        return found

    for path, inst in insts.items():
        cell = cells.get(inst.celltype)
        if cell is None or not cell.is_sequential or cell.clock_pin not in inst.pin2net:
            continue
        for src in sources(inst.pin2net[cell.clock_pin]):
            if src not in gates:
                e = other.setdefault(src, {"celltype": insts[src].celltype, "n_clocked": 0})
                e["n_clocked"] += 1

    n_series = sum(1 for r in rows.values() if r["parent"] is not None)
    return {
        "format": FORMAT,
        "n_nets": len(net_to_idx),
        "gates": [rows[g] for g in order],
        "ffs": ffs,
        "next_tables": next_tables,
        "several_gates": several,
        "unmodelled_gates": [{"inst": p, "celltype": insts[p].celltype, "type": r["type"]}
                             for p, r in sorted(unmodelled.items())],
        "other_clock_sources": dict(sorted(other.items())),
        "summary": {
            "gates": len(rows), "gates_in_series": n_series,
            "flip_flops_behind_gates": len(ffs),
            "gates_with_blackbox_pins": sum(1 for r in rows.values() if r["blackbox_pins"]),
            "cells_with_another_edge": sum(len(r["other_edge"]) for r in rows.values()),
            "cells_behind_several_gates": len(several),
            "cells_through_logic": sum(r["through_logic"] for r in rows.values()),
            "unmodelled_gates": len(unmodelled),
            "other_clock_sources": len(other)},
    }
