"""Three-state enable arcs and arcs without a delay, from the SDF to the records.

    FF a --a--> INV u1 --oe--> OE of PAD p0 --z--> D of FF c
    FF b --b----------------->  I of PAD p0

The pad's model gives the value of its net with the driver off (here a pull-up:
Z = OE ? I : 1).  The SDF gives its OE arc only as transitions to and from Z.
"""
import json
import os

import numpy as np
import pytest

from setfi.build import BuildError, BuildSpec, LibrarySpec, run_build
from setfi.campaign.logic import LogicSpec, run_logic
from setfi.campaign.run import CampaignSpec, run_campaign
from setfi.records import Records

CELLS = """
module INVX1 (A, ZN);
  input A;
  output ZN;
  assign ZN = ~A;
endmodule

module DFFX1 (D, CP, Q);
  input D, CP;
  output reg Q;
  always @(posedge CP) begin
    Q <= D;
  end
endmodule

module PADX1 (I, OE, Z);
  input I, OE;
  output Z;
  assign Z = PAD_FUNCTION;
endmodule
"""

NETLIST = """
module top (clk, in0, in1, out0);
  input clk, in0, in1;
  output out0;
  wire a, b, oe, z;
  DFFX1 ffa ( .D(in0), .CP(clk), .Q(a) );
  DFFX1 ffb ( .D(in1), .CP(clk), .Q(b) );
  INVX1 u1 ( .A(a), .ZN(oe) );
  PADX1 p0 ( .I(b), .OE(oe), .Z(z) );
  DFFX1 ffc ( .D(z), .CP(clk), .Q(out0) );
endmodule
"""

FF = """(DELAY (ABSOLUTE (IOPATH (posedge CP) Q (0.050) (0.050))))
    (TIMINGCHECK
      (SETUP (posedge D) (posedge CP) (0.030)) (SETUP (negedge D) (posedge CP) (0.030))
      (HOLD (posedge D) (posedge CP) (0.005)) (HOLD (negedge D) (posedge CP) (0.005)))"""

# 01 10 0Z Z1 1Z Z0: the transitions to and from Z only (ps: 0Z 21, Z1 33, 1Z 17, Z0 48)
OE_SIX = "(IOPATH OE Z () () (0.021:0.021:0.021) (0.033:0.033:0.033) (0.017:0.017:0.017) (0.048:0.048:0.048))"
PULL_UP = "OE ? I : 1'b1"


def sdf(oe_arc):
    cells = {"ffa": ("DFFX1", FF), "ffb": ("DFFX1", FF), "ffc": ("DFFX1", FF),
             "u1": ("INVX1", "(DELAY (ABSOLUTE (IOPATH A ZN (0.010) (0.010))))"),
             "p0": ("PADX1", f"(DELAY (ABSOLUTE (IOPATH I Z (0.040) (0.038)) {oe_arc}))")}
    body = "\n".join(f'  (CELL (CELLTYPE "{ct}") (INSTANCE {i})\n    {b})' for i, (ct, b) in cells.items())
    return f'(DELAYFILE (SDFVERSION "OVI 2.1") (DESIGN "top") (DIVIDER /) (TIMESCALE 1ns)\n{body}\n)\n'


def build(d, oe_arc=OE_SIX, pad=PULL_UP, **kw):
    os.makedirs(d, exist_ok=True)
    for name, text in (("cells.v", CELLS.replace("PAD_FUNCTION", pad)), ("top.v", NETLIST),
                       ("top.sdf", sdf(oe_arc))):
        with open(os.path.join(d, name), "w") as f:
            f.write(text)
    spec = BuildSpec(netlist=os.path.join(d, "top.v"), sdf=os.path.join(d, "top.sdf"), top="top",
                     out_dir=os.path.join(d, "build"),
                     library=LibrarySpec(behavioral_verilog=os.path.join(d, "cells.v")), **kw)
    return run_build(spec), spec.out_dir


def load(out, name):
    with open(os.path.join(out, name)) as f:
        return json.load(f)


def arc_delay(out, key):
    keys = load(out, "arc_index.json")["arc_idx_to_delay_key"]
    if key not in keys:
        return None
    dt = np.load(os.path.join(out, "delay_table.npz"))
    k = keys.index(key)
    return int(dt["default_rise_ps"][k]), int(dt["default_fall_ps"][k])


@pytest.mark.parametrize("pad, rise_fall", [
    (PULL_UP, (21, 48)),             # off -> 1: rise = 0Z (driver turns off), fall = Z0 (turns on)
    ("OE ? 1'b1 : I", (21, 48)),     # enable active low, pulled up: the same transitions
    ("OE ? I : 1'b0", (33, 17)),     # pulled down: rise = Z1 (turns on), fall = 1Z (turns off)
])
def test_three_state_arc_takes_rise_and_fall_from_the_cell_function(tmp_path, pad, rise_fall):
    man, out = build(str(tmp_path), pad=pad)
    assert arc_delay(out, "p0:OE->Z") == rise_fall
    assert arc_delay(out, "p0:I->Z") == (40, 38)
    assert man["sdf_reader_counts"]["iopath_three_state_values_only"] == 1
    assert man["warnings"] == [] and "arcs_without_delay" not in man


def test_three_value_form_gives_the_turn_off_value(tmp_path):
    # (01 10 Z): Z is the turn-off delay, 0Z and 1Z; pulled up, it is the rise
    _man, out = build(str(tmp_path), oe_arc="(IOPATH OE Z () () (0.030))")
    assert arc_delay(out, "p0:OE->Z")[0] == 30


def test_a_three_state_arc_without_a_released_value_has_no_delay(tmp_path):
    # Z = OE ^ I: no value of OE makes Z a constant, so the Z values cannot be placed
    with pytest.raises(BuildError, match=r"1 gate arc\(s\) inside injection cones have no SDF IOPATH "
                                         r"delay: PADX1 OE->Z"):
        build(str(tmp_path / "strict"), pad="OE ^ I")
    man, out = build(str(tmp_path / "allowed"), pad="OE ^ I", allow_missing_delays=True)
    assert arc_delay(out, "p0:OE->Z") is None
    assert man["sdf_notes"]["counters"]["sdf_three_state_arc_without_released_value"] == 1
    assert man["arcs_without_delay"]["by_cell_arc"] == {"PADX1:OE->Z": 1}


def test_every_function_arc_has_a_truth_table_without_an_sdf_delay(tmp_path):
    man, out = build(str(tmp_path), oe_arc="", allow_missing_delays=True)
    assert arc_delay(out, "p0:OE->Z") is None
    ent = load(out, "cell_arcs.json")["PADX1:OE->Z"]
    assert ent["vars_order"] == ["OE", "I"] and ent["truth"] == [1, 0, 1, 1]   # index = OE + 2*I
    assert man["arcs_without_delay"]["n_arcs"] == 1
    assert "pulses do not pass these arcs" in man["warnings"][0]


# ---------------------------------------------------------------------------- inject
def states(d, out):
    """cycle 1: a = 0 (oe = 1), b = 0, so z = 0; cycle 2: b = 1, so z = 1."""
    idx = load(out, "net_index.json")["net_to_idx"]
    os.makedirs(d, exist_ok=True)
    for cyc, vals in {1: dict(a=0, b=0, oe=1, z=0), 2: dict(a=0, b=1, oe=1, z=1)}.items():
        with open(os.path.join(d, f"cycle_{cyc}.hex"), "w") as f:
            f.write(f"{sum(v << idx[n] for n, v in vals.items()):x}\n")


def inject(tmp_path, out, name="inject", logs=None):
    states(str(tmp_path / "states"), out)
    spec = CampaignSpec(stage_dir=out, state_dir=str(tmp_path / "states"), cycles=[1, 2],
                        out_dir=str(tmp_path / name), clock_period_ps=1000, pulse_widths_ps=[100],
                        sites=["oe"])
    idx = run_campaign(spec, log=(logs.append if logs is not None else (lambda *a: None)))
    rec = Records.open(str(tmp_path / name))
    c = rec.load_all()
    rows = sorted((int(c["cycle"][i]), int(c["base"][i]), int(c["t_enter"][i]), int(c["t_exit"][i]))
                  for i in range(c["cycle"].size))
    return rows, idx


def test_a_pulse_on_the_enable_passes_with_the_z_transition_delays(tmp_path):
    _man, out = build(str(tmp_path))
    rows, idx = inject(tmp_path, out)
    # cycle 1 (I = 0): OE 1 -> 0 at 0 releases the pad, z rises after 0Z = 21; OE 0 -> 1 at
    # 100 drives it again, z falls after Z0 = 48.  Cycle 2 (I = 1): z stays 1.
    assert rows == [(1, 0, 21, 148)]
    assert "arcs_without_delay" not in idx["counts"]


@pytest.mark.parametrize("engine", ["fast", "reference"])
def test_arcs_without_delay_stop_pulses_and_are_counted(tmp_path, monkeypatch, engine):
    if engine == "reference":
        monkeypatch.setenv("SETFI_REFERENCE_ENGINE", "1")
    _man, out = build(str(tmp_path), oe_arc="", allow_missing_delays=True)
    logs = []
    rows, idx = inject(tmp_path, out, logs=logs)
    assert rows == []
    assert idx["counts"]["arcs_without_delay"] == 1
    assert any("1 gate arc(s) in the cones of 1 site(s) have no delay" in m for m in logs)


@pytest.mark.parametrize("keep_other_arc", [True, False])
def test_a_build_without_the_truth_table_of_the_arc_does_not_stop_inject(tmp_path, keep_other_arc):
    # cell_arcs.json as a build writes it that has entries only for the arcs with a delay
    _man, out = build(str(tmp_path), oe_arc="", allow_missing_delays=True)
    arcs = load(out, "cell_arcs.json")
    del arcs["PADX1:OE->Z"]
    if not keep_other_arc:
        del arcs["PADX1:I->Z"]
    with open(os.path.join(out, "cell_arcs.json"), "w") as f:
        json.dump(arcs, f)
    rows, idx = inject(tmp_path, out)
    assert rows == [] and idx["counts"]["arcs_without_delay"] == 1
    # logic-only: no delays, so the change of oe reaches z when the cell function is known
    pt = run_logic(LogicSpec(stage_dir=out, state_dir=str(tmp_path / "states"), cycles=[1, 2],
                             sites=["oe"]), log=lambda *a: None)
    assert [int(c) for c in pt.pattern_cycle] == ([1] if keep_other_arc else [])
