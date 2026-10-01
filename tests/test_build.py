"""Hermetic tests of the build stage: a hand-written behavioural library, netlist and
SDF (no PDK), small enough that every expected edge, site and cone is written out."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from setfi.build import BuildError, BuildSpec, LibrarySpec, run_build
from setfi.build.netlist import elaborate, parse_modules, strip_comments

LIB = """
// assign-style behavioural library
module INVX1 (A, ZN);
  input A;
  output ZN;
  assign ZN = ~A;
endmodule

module ND2X1 (A1, A2, ZN);
  input A1, A2;
  output ZN;
  wire n;
  assign n = A1 & A2;   /* internal wire, substituted */
  assign ZN = !n;
endmodule

module DFFX1 (D, CP, Q);
  input D, CP;
  output reg Q;
  always @(posedge CP) begin
    Q <= D;
  end
endmodule

module TIEHX1 (Z);
  output Z;
  assign Z = 1'b1;
endmodule

module CKLNQX1 (TE, E, CP, Q);
  input TE, E, CP;
  output Q;
  reg l;
  wire en;
  assign en = TE | E;
  always @(*) begin
    if (!CP) begin
      l <= en;
    end
  end
  assign Q = l & CP;
endmodule

module DFNX1 (D, CPN, Q);
  input D, CPN;
  output reg Q;
  wire CP;
  assign CP = !CPN;
  always @(posedge CP) begin
    Q <= D;
  end
endmodule

module BADX1 (A, B, ZN);
  input A, B;
  output ZN;
  assign ZN = A $ B;
endmodule

module FILLX1;
endmodule
"""

# A wrapper module whose vector port is bound to a concatenation: the child's a[1] is
# the concatenation's MSB (n1), not "n1, n0 bit 1".
NETLIST = """
module sub (a, y);
  input [1:0] a;
  output y;
  INVX1 ui ( .A(a[1]), .ZN(y) );
endmodule

module top (clk, in0, out0);
  input clk, in0;
  output out0;
  wire q0, n0, n1, n2, n3, q1, tie;
  DFFX1 ff0 ( .D(in0), .CP(clk), .Q(q0) );
  INVX1 u0 ( .A(q0), .ZN(n0) );
  INVX1 u1 ( .A(n0), .ZN(n1) );
  sub us ( .a({n1, n0}), .y(n2) );
  ND2X1 u3 ( .A1(n2), .A2(n0), .ZN(n3) );
  DFFX1 ff1 ( .D(n3), .CP(clk), .Q(q1) );
  INVX1 u4 ( .A(q1), .ZN(out0) );
  TIEHX1 t0 ( .Z(tie) );
EXTRA
endmodule
"""

DFF_TC = """(TIMINGCHECK
      (SETUP (posedge D) (posedge CP) (0.030:0.030:0.030))
      (SETUP (negedge D) (posedge CP) (0.040:0.040:0.040))
      (HOLD (posedge D) (posedge CP) (-0.010:-0.012:-0.012))
      (HOLD (negedge D) (posedge CP) (0.005:0.006:0.006)))"""
DFF_TC_INNOVUS = """(TIMINGCHECK
      (SETUPHOLD (posedge D) (posedge CP) (0.030::0.035) (-0.010::-0.012))
      (SETUPHOLD (negedge D) (posedge CP) (0.040::0.045) (0.005::0.006))
      (WIDTH (posedge CP) (0.100::0.100)))"""
INV_DELAY = "(DELAY (ABSOLUTE (IOPATH A ZN (0.010:0.012:0.012) (0.008:0.009:0.009))))"
FF_DELAY = "(DELAY (ABSOLUTE (IOPATH (posedge CP) Q (0.050:0.050:0.050) (0.060:0.060:0.060))))"

BASE_CELLS = {
    "ff0": ("DFFX1", FF_DELAY + "\n    " + DFF_TC),
    "u0": ("INVX1", INV_DELAY),
    "u1": ("INVX1", INV_DELAY),
    "us/ui": ("INVX1", INV_DELAY),
    "u3": ("ND2X1", """(DELAY (ABSOLUTE
      (IOPATH A1 ZN (0.020:0.021:0.021) (0.015:0.016:0.016))
      (COND A1 == 1'b1 (IOPATH A2 ZN (0.018:0.019:0.019) (0.013:0.014:0.014)))))"""),
    "ff1": ("DFFX1", FF_DELAY + "\n    " + DFF_TC),
    "u4": ("INVX1", INV_DELAY),
}

INTERCONNECT = """
    (INTERCONNECT u0/ZN u1/A (0.002:0.002:0.002) (0.003:0.003:0.003))
    (INTERCONNECT u3/ZN ff1/D (0.001:0.001:0.001))
    (INTERCONNECT u4/ZN out0 (0.000:0.000:0.000))"""


def make_sdf(cells, design="top", version="OVI 2.1", interconnect=INTERCONNECT):
    parts = [f"""(DELAYFILE
  (SDFVERSION "{version}")
  (DESIGN "{design}")
  (DATE "today")
  (VENDOR "test")
  (PROGRAM "handwritten")
  (VERSION "1")
  (DIVIDER /)
  (TIMESCALE 1ns)"""]
    if interconnect:
        parts.append(f"""  (CELL (CELLTYPE "{design}") (INSTANCE)
    (DELAY (ABSOLUTE {interconnect})))""")
    for inst, (celltype, body) in cells.items():
        parts.append(f'  (CELL (CELLTYPE "{celltype}") (INSTANCE {inst})\n    {body})')
    parts.append(")")
    return "\n".join(parts) + "\n"


def write_case(tmp_path, *, extra="", cells=None, sdf=None, lib=LIB, **spec_kw):
    (tmp_path / "cells.v").write_text(lib)
    (tmp_path / "top.v").write_text(NETLIST.replace("EXTRA", extra))
    (tmp_path / "top.sdf").write_text(sdf if sdf is not None
                                      else make_sdf(cells if cells is not None else BASE_CELLS))
    lib_kw = {k[4:]: spec_kw.pop(k) for k in list(spec_kw) if k.startswith("lib_")}
    return BuildSpec(netlist=str(tmp_path / "top.v"), sdf=str(tmp_path / "top.sdf"),
                     top="top", out_dir=str(tmp_path / "out"),
                     library=LibrarySpec(behavioral_verilog=str(tmp_path / "cells.v"),
                                         **lib_kw),
                     **spec_kw)


def load(out, name):
    with open(os.path.join(out, name)) as f:
        return json.load(f)


def skeleton(out):
    with open(os.path.join(out, "site_cones.jsonl")) as f:
        return [json.loads(line) for line in f]


# ----------------------------------------------------------------------------
def test_small_design_graph(tmp_path):
    spec = write_case(tmp_path)
    man = run_build(spec)
    out = spec.out_dir

    ni = load(out, "net_index.json")
    assert list(ni["net_to_idx"]) == ["in0", "n0", "n1", "n2", "n3", "out0", "q0", "q1", "tie"]
    idx = ni["net_to_idx"]

    edges = load(out, "edges.json")
    assert [(e["src_net"], e["dst_net"], e["delay_key"]) for e in edges] == [
        ("q0", "n0", "u0:A->ZN"),
        ("n0", "n1", "u1:A->ZN"),
        ("n1", "n2", "us/ui:A->ZN"),      # a[1] of {n1, n0} is n1
        ("n2", "n3", "u3:A1->ZN"),
        ("n0", "n3", "u3:A2->ZN"),
        ("q1", "out0", "u4:A->ZN"),
    ]
    assert edges[3]["mask_key"] == "ND2X1:A1->ZN"
    assert [e["eid"] for e in edges] == list(range(6))

    p2n = load(out, "pin_to_net.json")
    assert p2n["us/ui.A"] == "n1" and p2n["us/ui.ZN"] == "n2" and p2n["us.a"] == "{n1, n0}"

    # Sites: the forward region from FF outputs (q0 -> n0 ...), plus FF D nets (in0,
    # n3), minus FF Q nets.  The cone stops at FF D nets.
    sk = skeleton(out)
    inv = {v: k for k, v in idx.items()}
    got = {inv[r["src_net_idx"]]: r for r in sk}
    assert list(got) == ["in0", "n0", "n1", "n2", "n3"]
    n = lambda names: [idx[x] for x in names]  # noqa: E731
    assert got["n0"]["cone_topo_net_idxs"] == n(["n0", "n1", "n2", "n3"])
    assert got["n0"]["cone_adj_idx"] == [[idx["n0"], [1, 4]], [idx["n1"], [2]], [idx["n2"], [3]]]
    assert got["n0"]["has_reconv"] is True           # n0 -> n3 directly and via n1, n2
    assert got["n1"]["has_reconv"] is False
    assert got["n1"]["cone_topo_net_idxs"] == n(["n1", "n2", "n3"])
    for s in ("n0", "n1", "n2", "n3"):
        assert got[s]["reachable_dnet_idxs"] == n(["n3"])
    # A site that IS an FF's D net captures itself (and nothing else).
    assert got["n3"]["cone_topo_net_idxs"] == n(["n3"]) and got["n3"]["cone_adj_idx"] == []
    assert got["in0"]["reachable_dnet_idxs"] == n(["in0"])

    ff = load(out, "ff_index.json")
    info = ff["ffid_to_info"]
    assert {k: (v["ff_inst"], v["d_pin"], v["d_net"]) for k, v in info.items()} == {
        "0": ("ff0", "D", "in0"), "1": ("ff1", "D", "n3")}
    assert ff["dnet_to_ffids"] == {str(idx["in0"]): [0], str(idx["n3"]): [1]}
    assert ff["_meta"]["observe_filter"]["n_kept"] == 2

    assert man["checks"]["parse_integrity"]["status"] == "pass"
    assert man["netlist"]["sdf_crosscheck"]["ok"]
    assert man["netlist"]["sdf_crosscheck"]["n_untimed_leaves"] == 1    # the tie cell
    assert man["graph"]["self_check_samples_ok"] == 5
    assert set(man["artifacts"]) == {
        "net_index.json", "site_cones.jsonl", "edges.json", "pin_to_net.json",
        "ff_index.json", "cell_arcs.json", "timing_checks.json",
        "delay_table.npz", "arc_index.json", "interconnect.json"}
    assert man["warnings"] == []


def test_small_design_timing(tmp_path):
    spec = write_case(tmp_path)
    run_build(spec)
    out = spec.out_dir

    # timing checks: per-FF capture thresholds, the late rail by default.
    b4 = load(out, "timing_checks.json")
    assert b4["version"] == 3 and b4["sdf_delay_rail"] == "late"
    ff1 = b4["index"]["1"]                          # inst index: ff0, ff1, t0, u0, ...
    assert ff1["meta"] == {"ff_inst": "ff1", "inst_idx": 1, "celltype": "DFFX1"}
    rise = ff1["checks"]["posedge:D"]["posedge:CP"]
    fall = ff1["checks"]["negedge:D"]["posedge:CP"]
    assert rise["SETUP"][0]["value"] == 0.030 and rise["HOLD"][0]["value"] == -0.012
    assert fall["SETUP"][0]["value"] == 0.040 and fall["HOLD"][0]["value"] == 0.006
    assert rise["SETUP"][0]["data_net_idx"] == load(out, "net_index.json")["net_to_idx"]["n3"]

    # cell arcs: the ND2 function through the internal wire, with its side pin and truth table.
    b3 = load(out, "cell_arcs.json")
    nd = b3["ND2X1:A2->ZN"]
    assert nd["kind"] == "comb" and nd["side_pins"] == ["A1"] and nd["vars_order"] == ["A2", "A1"]
    assert nd["truth"] == [1, 1, 1, 0]               # index = A2 + 2*A1
    assert b3["INVX1:A->ZN"]["truth"] == [1, 0]
    assert "DFFX1:CP->Q" not in b3                   # FF outputs are cone boundaries

    # Delay table: integer ps on the late rail; the COND arc gets a pattern keyed on A1.
    ai = load(out, "arc_index.json")
    keys = ai["arc_idx_to_delay_key"]
    assert keys == ["ff0:CP->Q", "ff1:CP->Q", "u0:A->ZN", "u1:A->ZN", "u3:A1->ZN",
                    "u3:A2->ZN", "u4:A->ZN", "us/ui:A->ZN"]
    dt = np.load(os.path.join(out, "delay_table.npz"))
    k = keys.index("u0:A->ZN")
    assert (dt["default_rise_ps"][k], dt["default_fall_ps"][k]) == (12, 9)
    k = keys.index("u3:A2->ZN")
    lo, hi = dt["patt_ptr"][k], dt["patt_ptr"][k + 1]
    assert list(dt["patt_mask"][lo:hi]) == [1, 1] and list(dt["patt_valbits"][lo:hi]) == [1, 1]
    assert list(dt["patt_dt_ps"][lo:hi]) == [19, 14] and list(dt["patt_is_rise"][lo:hi]) == [1, 0]

    # interconnect: only the wire into a combinational input is transportable.
    b5 = load(out, "interconnect.json")
    assert b5["entries"] == [{"net": "n0", "net_idx": 1, "sink_inst": "u1", "sink_pin": "A",
                              "sink_celltype": "INVX1", "rise_ps": 2, "fall_ps": 3}]
    acc = b5["row_account"]
    assert (acc["total"], acc["sink_comb_input"], acc["sink_seq_data_pin"],
            acc["sink_top_port"]) == (3, 1, 1, 1)


def test_early_rail_and_innovus_dialect(tmp_path):
    cells = dict(BASE_CELLS)
    for ff in ("ff0", "ff1"):
        cells[ff] = ("DFFX1", FF_DELAY + "\n    " + DFF_TC_INNOVUS)
    spec = write_case(tmp_path, sdf=make_sdf(cells, version="3.0"), sdf_rail="early")
    man = run_build(spec)
    b4 = load(spec.out_dir, "timing_checks.json")
    rise = b4["index"]["1"]["checks"]["posedge:D"]["posedge:CP"]
    # SETUPHOLD split into SETUP + HOLD; (a::b) keeps typ empty; early = first field.
    assert rise["SETUP"][0]["typ"] is None and rise["SETUP"][0]["value"] == 0.030
    assert rise["HOLD"][0]["value"] == -0.010 and rise["HOLD"][0]["rail"] == "early"
    assert man["sdf_reader_counts"]["timing_checks_not_modelled:WIDTH"] == 2
    dt = np.load(os.path.join(spec.out_dir, "delay_table.npz"))
    keys = load(spec.out_dir, "arc_index.json")["arc_idx_to_delay_key"]
    assert dt["default_rise_ps"][keys.index("u0:A->ZN")] == 10


def test_observe_filter_drops_clock_gate(tmp_path):
    cells = dict(BASE_CELLS)
    cells["icg0"] = ("CKLNQX1", "(DELAY (ABSOLUTE (IOPATH CP Q (0.02:0.02:0.02))))")
    spec = write_case(tmp_path, cells=cells,
                      extra="  CKLNQX1 icg0 ( .TE(tie), .E(n1), .CP(clk), .Q(gclk) );")
    man = run_build(spec)
    ff = load(spec.out_dir, "ff_index.json")
    # icg0 is recognised as sequential (clock CP, "D" = E) and gets an ffid, but is not
    # observed; the kept ffids are not renumbered.
    assert {v["ff_inst"] for v in ff["ffid_to_info"].values()} == {"ff0", "ff1"}
    assert set(ff["ffid_to_info"]) == {"0", "1"}
    meta = ff["_meta"]["observe_filter"]
    # with the default configuration the clock gate is dropped because its data pin is E
    assert meta["n_input"] == 3 and meta["n_removed"] == {"excluded_cell": 0, "non_data_pin": 1,
                                                         "excluded_instance": 0}
    assert meta["predicates"] == ["d_pin == 'D'"]
    assert man["observed_ffs"]["removed_examples"] == ["icg0"]


def test_cell_without_parsable_function_raises(tmp_path):
    cells = dict(BASE_CELLS)
    cells["b0"] = ("BADX1", "(DELAY (ABSOLUTE (IOPATH A ZN (0.01:0.01:0.01))))")
    spec = write_case(tmp_path, cells=cells, extra="  BADX1 b0 ( .A(n0), .B(n1), .ZN(nb) );")
    with pytest.raises(BuildError, match=r"BADX1: output ZN has no parsable function"):
        run_build(spec)


def test_stateful_cell_needs_its_clock_pin_declared(tmp_path):
    cells = dict(BASE_CELLS)
    cells["fn0"] = ("DFNX1", "(DELAY (ABSOLUTE (IOPATH (negedge CPN) Q (0.05:0.05:0.05))))")
    extra = "  DFNX1 fn0 ( .D(n2), .CPN(clk), .Q(qn) );"
    spec = write_case(tmp_path, cells=cells, extra=extra)
    with pytest.raises(BuildError, match=r"DFNX1: its model holds state"):
        run_build(spec)
    spec = write_case(tmp_path, cells=cells, extra=extra,
                      lib_clock_pin_names=("CP", "CK", "CLK", "CLKN", "G", "E", "CPN"))
    run_build(spec)
    ff = load(spec.out_dir, "ff_index.json")
    assert "fn0" in {v["ff_inst"] for v in ff["ffid_to_info"].values()}


def test_unknown_cell_type_raises_unless_blackbox(tmp_path):
    extra = "  SRAMX m0 ( .A(n0), .Q(mq) );\n  INVX1 u5 ( .A(mq), .ZN(n5) );\n" \
            "  DFFX1 ff2 ( .D(n5), .CP(clk), .Q(q2) );"
    cells = dict(BASE_CELLS)
    cells["u5"] = ("INVX1", INV_DELAY)
    cells["ff2"] = ("DFFX1", FF_DELAY + "\n    " + DFF_TC)
    spec = write_case(tmp_path, cells=cells, extra=extra)
    with pytest.raises(BuildError, match=r"cell type 'SRAMX'"):
        run_build(spec)
    spec = write_case(tmp_path, cells=cells, extra=extra, lib_blackbox_cells=("SRAM.*",))
    man = run_build(spec)
    idx = load(spec.out_dir, "net_index.json")["net_to_idx"]
    sites = {r["src_net_idx"] for r in skeleton(spec.out_dir)}
    # The macro output seeds the forward region (its fanout n5 is a site) but is not
    # itself injectable.
    assert idx["n5"] in sites and idx["mq"] not in sites
    assert man["netlist"]["n_blackboxes"] == 1


def test_sdf_netlist_mismatch(tmp_path):
    cells = {k: v for k, v in BASE_CELLS.items() if k != "u4"}
    spec = write_case(tmp_path, cells=cells)
    with pytest.raises(BuildError, match=r"1 netlist instance\(s\) have no SDF entry, e.g. \['u4'\]"):
        run_build(spec)
    spec = write_case(tmp_path, cells=cells, strict_sdf_crosscheck=False)
    man = run_build(spec)
    assert any("disagree" in w for w in man["warnings"])

    spec = write_case(tmp_path, sdf=make_sdf(BASE_CELLS, design="other_top"))
    with pytest.raises(BuildError, match=r"written for the design 'other_top'"):
        run_build(spec)


def test_port_width_mismatch_is_refused(tmp_path):
    spec = write_case(tmp_path)
    (tmp_path / "top.v").write_text(
        NETLIST.replace("EXTRA", "").replace(".a({n1, n0})", ".a({n1, n0, n2})"))
    with pytest.raises(BuildError, match=r"declared \[1:0\] \(2 bits\) but the parent "
                                         r"connects a 3-bit"):
        run_build(spec)


def test_unsupported_netlist_shapes_are_refused(tmp_path):
    # Positional connections would parse to an empty pin map.
    spec = write_case(tmp_path)
    (tmp_path / "top.v").write_text(NETLIST.replace("EXTRA", "").replace(
        "INVX1 u4 ( .A(q1), .ZN(out0) );", "INVX1 u4 ( q1, out0 );"))
    with pytest.raises(BuildError, match=r"\['u4'\] are connected by position"):
        run_build(spec)
    # ANSI headers would parse to a module without ports.
    (tmp_path / "top.v").write_text(NETLIST.replace("EXTRA", "").replace(
        "module sub (a, y);\n  input [1:0] a;\n  output y;",
        "module sub (input [1:0] a, output y);"))
    with pytest.raises(BuildError, match=r"module 'sub' \(at us\) declares its port "
                                         r"directions in the header"):
        run_build(spec)


def test_failed_build_leaves_no_substrate(tmp_path):
    spec = write_case(tmp_path)
    run_build(spec)
    assert os.path.exists(os.path.join(spec.out_dir, "edges.json"))
    cells = {k: v for k, v in BASE_CELLS.items() if k != "u4"}
    (tmp_path / "top.sdf").write_text(make_sdf(cells))
    with pytest.raises(BuildError):
        run_build(spec)
    left = set(os.listdir(spec.out_dir)) - {"intermediate"}
    assert left == set()


def test_intermediates_and_stale_files(tmp_path):
    spec = write_case(tmp_path, keep_intermediates=True)
    run_build(spec)
    inter = os.path.join(spec.out_dir, "intermediate")
    assert {"cone_db.jsonl", "superedges.json", "net_entry.json", "inst_index.json",
            "arc_delays.json"} <= set(os.listdir(inter))
    # Rebuilding from an SDF without INTERCONNECT must not leave the old interconnect behind.
    spec2 = write_case(tmp_path, sdf=make_sdf(BASE_CELLS, interconnect=""))
    man = run_build(spec2)
    assert not os.path.exists(os.path.join(spec.out_dir, "interconnect.json"))
    assert "interconnect.json" not in man["artifacts"]


def test_exclude_d_from_src(tmp_path):
    spec = write_case(tmp_path, exclude_d_from_src=True)
    run_build(spec)
    idx = load(spec.out_dir, "net_index.json")["net_to_idx"]
    sites = {r["src_net_idx"] for r in skeleton(spec.out_dir)}
    assert sites == {idx["n0"], idx["n1"], idx["n2"]}


def test_spec_from_dict(tmp_path):
    spec = write_case(tmp_path)
    d = spec.to_dict()
    lib = d.pop("library")
    again = BuildSpec.from_dict(d, lib)
    assert again.to_dict() == spec.to_dict()
    with pytest.raises(BuildError, match="unknown \\[build\\] key"):
        BuildSpec.from_dict({**d, "use_hierarchy": True}, lib)
    with pytest.raises(BuildError, match="unknown \\[library\\] key"):
        BuildSpec.from_dict(d, {**lib, "lib_v": "x"})
    bad = BuildSpec.from_dict({**d, "sdf_rail": "typ"}, lib)
    with pytest.raises(BuildError, match="sdf_rail"):
        run_build(bad)


def test_parser_units():
    # A comment opener inside another comment or a string is not a comment opener.
    assert strip_comments('a // x /* y\nb /* // */ c "d // e"') == 'a \nb  c "d // e"'
    mods = parse_modules(NETLIST.replace("EXTRA", ""))
    lib = {"INVX1", "ND2X1", "DFFX1", "TIEHX1"}
    elab = elaborate(mods, "top", lib, lambda c: False)
    assert list(elab.leaves) == ["ff0", "u0", "u1", "us", "us/ui", "u3", "ff1", "u4", "t0"]
    assert elab.leaves["us/ui"].pin2net == {"A": "n1", "ZN": "n2"}
    assert mods["top"].port_dirs == {"clk": "input", "in0": "input", "out0": "output"}


# ---------------------------------------------------------------- netlist `assign` aliases
ALIAS_NETLIST = """
module fwd (a, y, z, c);
  input [1:0] a;
  output [1:0] y;
  output z;
  output [2:0] c;
  wire n;
  INVX1 ui ( .A(a[0]), .ZN(n) );
  assign z = n;                 // output port driven through an alias
  assign y = {a[0], a[1]};      // bus alias with a swap
  assign c = 2'b10;             // tied (and zero-extended to 3 bits)
endmodule

module top (x0, x1, o0, o1);
  input x0, x1;
  output o0, o1;
  wire [1:0] w;
  wire zz;
  wire [2:0] cc;
  fwd f ( .a({x1, x0}), .y(w), .z(zz), .c(cc) );
  INVX1 r0 ( .A(zz), .ZN(o0) );
  ND2X1 r1 ( .A1(w[1]), .A2(cc[1]), .ZN(o1) );
endmodule
"""


def _elab():
    mods = parse_modules(ALIAS_NETLIST)
    return elaborate(mods, "top", {"INVX1", "ND2X1"}, lambda c: False)


def test_assign_alias_connects_reader_to_driver():
    el = _elab()
    assert el.leaves["f/ui"].pin2net["ZN"] == "f/n"
    assert el.leaves["r0"].pin2net["A"] == "f/n"          # zz -> f/n
    assert el.leaves["r1"].pin2net["A1"] == "x0"          # w[1] = a[0] = x0
    assert el.leaves["r1"].pin2net["A2"] == "cc[1]"       # tied to 1: stays undriven
    assert (el.n_alias_bits, el.n_tied_bits) == (3, 3)


def test_assign_with_logic_is_refused():
    mods = parse_modules(ALIAS_NETLIST.replace("assign z = n;", "assign z = ~n;"))
    with pytest.raises(BuildError, match="not a net alias"):
        elaborate(mods, "top", {"INVX1", "ND2X1"}, lambda c: False)


def test_ansi_headers_and_always_without_begin(tmp_path):
    """The same design with the library written in ANSI style and `always` blocks
    without begin/end builds to the same graph."""
    ansi = (LIB
            .replace("module INVX1 (A, ZN);\n  input A;\n  output ZN;",
                     "module INVX1 (input A, output ZN);")
            .replace("module DFFX1 (D, CP, Q);\n  input D, CP;\n  output reg Q;\n"
                     "  always @(posedge CP) begin\n    Q <= D;\n  end",
                     "module DFFX1 (input D, input CP, output reg Q);\n"
                     "  always @(posedge CP) Q <= D;"))
    assert "module INVX1 (input A" in ansi and "always @(posedge CP) Q <= D;" in ansi
    (tmp_path / "ref").mkdir()
    (tmp_path / "new").mkdir()
    ref = write_case(tmp_path / "ref")
    new = write_case(tmp_path / "new", lib=ansi)
    run_build(ref)
    run_build(new)
    for name in ("edges.json", "ff_index.json", "site_cones.jsonl", "cell_arcs.json"):
        assert (tmp_path / "ref" / "out" / name).read_text() == (tmp_path / "new" / "out" / name).read_text()


def test_sdf_timescale_and_divider_are_converted(tmp_path):
    """The same SDF written in ps with `.` as divider builds to the same timing data."""
    import re
    from decimal import Decimal
    (tmp_path / "ns").mkdir()
    (tmp_path / "ps").mkdir()
    ref = write_case(tmp_path / "ns")
    text = (tmp_path / "ns" / "top.sdf").read_text()
    head, body = text.split("(CELL", 1)
    head = re.sub(r"\(TIMESCALE[^)]*\)", "(TIMESCALE 1ps)", head)
    head = re.sub(r"\(DIVIDER\s*/\s*\)", "(DIVIDER .)", head)
    body = re.sub(r"-?\d+\.\d+", lambda m: str(Decimal(m.group(0)) * 1000), body)
    body = re.sub(r"\(INSTANCE ([^)]*)\)", lambda m: "(INSTANCE " + m.group(1).replace("/", ".") + ")", body)
    new = write_case(tmp_path / "ps", sdf=head + "(CELL" + body)
    assert "TIMESCALE 1ps" in (tmp_path / "ps" / "top.sdf").read_text()
    run_build(ref)
    run_build(new)
    for name in ("timing_checks.json", "arc_index.json"):
        assert (tmp_path / "ns" / "out" / name).read_text() == (tmp_path / "ps" / "out" / name).read_text()
    a = np.load(tmp_path / "ns" / "out" / "delay_table.npz")
    b = np.load(tmp_path / "ps" / "out" / "delay_table.npz")
    assert all(np.array_equal(a[k], b[k]) for k in a.files)


HIER_CELLS = """
module INV (input A, output ZN); assign ZN = !A; endmodule
module ND2 (input A1, input A2, output ZN); assign ZN = !(A1 & A2); endmodule
module DFF (input D, input CP, output reg Q); always @(posedge CP) Q <= D; endmodule
"""
FLAT = """
module top (clk, q2);
  input clk; output q2;
  wire q0, q1, n1, n2;
  DFF r0 (.D(n2), .CP(clk), .Q(q0));
  DFF r1 (.D(q0), .CP(clk), .Q(q1));
  ND2 u1 (.A1(q0), .A2(q1), .ZN(n1));
  INV u2 (.A(n1), .ZN(n2));
  DFF r2 (.D(n2), .CP(clk), .Q(q2));
endmodule
"""
HIER = """
module blk (a, b, y);
  input a, b; output y;
  wire n1;
  ND2 u1 (.A1(a), .A2(b), .ZN(n1));
  INV u2 (.A(n1), .ZN(y));
endmodule
module top (clk, q2);
  input clk; output q2;
  wire q0, q1, n2;
  DFF r0 (.D(n2), .CP(clk), .Q(q0));
  DFF r1 (.D(q0), .CP(clk), .Q(q1));
  blk b (.a(q0), .b(q1), .y(n2));
  DFF r2 (.D(n2), .CP(clk), .Q(q2));
endmodule
"""


def _hier_sdf(prefix):
    ff = ('(CELL (CELLTYPE "DFF") (INSTANCE {n}) (DELAY (ABSOLUTE (IOPATH (posedge CP) Q (0.05))))'
          ' (TIMINGCHECK (SETUP D (posedge CP) (0.02)) (HOLD D (posedge CP) (0.01))))\n')
    return ('(DELAYFILE (SDFVERSION "3.0") (DESIGN "top") (TIMESCALE 1ns) (DIVIDER /)\n'
            + "".join(ff.format(n=n) for n in ("r0", "r1", "r2"))
            + f'(CELL (CELLTYPE "ND2") (INSTANCE {prefix}u1) (DELAY (ABSOLUTE '
              '(IOPATH A1 ZN (0.02)) (IOPATH A2 ZN (0.02)))))\n'
            + f'(CELL (CELLTYPE "INV") (INSTANCE {prefix}u2) (DELAY (ABSOLUTE (IOPATH A ZN (0.01)))))\n)\n')


def test_hierarchy_does_not_change_the_sites(tmp_path):
    """A flip-flop output wired to another flip-flop's data pin is a site, whether or
    not it also feeds a sub-module."""
    sites = {}
    for name, text, prefix in (("flat", FLAT, ""), ("hier", HIER, "b/")):
        d = tmp_path / name
        d.mkdir()
        (d / "cells.v").write_text(HIER_CELLS)
        (d / "net.v").write_text(text)
        (d / "net.sdf").write_text(_hier_sdf(prefix))
        man = run_build(BuildSpec(netlist=str(d / "net.v"), sdf=str(d / "net.sdf"), top="top",
                                  out_dir=str(d / "out"),
                                  library=LibrarySpec(behavioral_verilog=str(d / "cells.v"))))
        idx = json.loads((d / "out" / "net_index.json").read_text())["idx_to_net"]
        sites[name] = sorted(idx[str(json.loads(l)["src_net_idx"])]
                             for l in (d / "out" / "site_cones.jsonl").read_text().splitlines())
        assert not man["warnings"]
    assert sites["flat"] == ["n1", "n2", "q0"]
    assert sites["hier"] == ["b/n1", "n2", "q0"]
