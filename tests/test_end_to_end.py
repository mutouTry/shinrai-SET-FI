"""`setfi run` on a 2-bit counter with Icarus Verilog (skipped if it is not installed)."""
import json
import os
import shutil

import pytest

from setfi import cli

pytestmark = pytest.mark.skipif(shutil.which("iverilog") is None or shutil.which("vvp") is None,
                                reason="Icarus Verilog not installed")

CELLS = """
module INV (input A, output ZN);
  assign ZN = !A;
endmodule
module XOR2 (input A, input B, output Z);
  assign Z = A ^ B;
endmodule
module DFFR (input D, input CK, input RN, output reg Q);
  always @(posedge CK or negedge RN)
    if (!RN) Q <= 1'b0;
    else Q <= D;
endmodule
"""

NETLIST = """
module cnt2 (clk, rst_n, q0, q1);
  input clk, rst_n;
  output q0, q1;
  wire n0, n1;
  DFFR f0 ( .D(n0), .CK(clk), .RN(rst_n), .Q(q0) );
  DFFR f1 ( .D(n1), .CK(clk), .RN(rst_n), .Q(q1) );
  INV u0 ( .A(q0), .ZN(n0) );
  XOR2 u1 ( .A(q0), .B(q1), .Z(n1) );
endmodule
"""


def sdf(delay_ns="0.030"):
    ff = """  (CELL (CELLTYPE "DFFR") (INSTANCE {n})
    (DELAY (ABSOLUTE (IOPATH (posedge CK) Q (0.050) (0.050))))
    (TIMINGCHECK
      (SETUP (posedge D) (posedge CK) (0.020)) (SETUP (negedge D) (posedge CK) (0.020))
      (HOLD (posedge D) (posedge CK) (0.005)) (HOLD (negedge D) (posedge CK) (0.005))))
"""
    return ("""(DELAYFILE (SDFVERSION "3.0") (DESIGN "cnt2") (TIMESCALE 1ns) (DIVIDER /)
""" + ff.format(n="f0") + ff.format(n="f1") + f"""  (CELL (CELLTYPE "INV") (INSTANCE u0)
    (DELAY (ABSOLUTE (IOPATH A ZN ({delay_ns}) ({delay_ns})))))
  (CELL (CELLTYPE "XOR2") (INSTANCE u1)
    (DELAY (ABSOLUTE (IOPATH A Z ({delay_ns}) ({delay_ns})) (IOPATH B Z ({delay_ns}) ({delay_ns})))))
)
""")


TB = """
`timescale 1ns/1ps
module tb;
  reg clk = 0;
  reg rst_n = RESET_INIT;
  wire q0, q1;
  cnt2 dut ( .clk(clk), .rst_n(rst_n), .q0(q0), .q1(q1) );
  always #1 clk = ~clk;
  initial begin
    #4 rst_n = 1;
    #200;
    $display("RESULT: PASS");
    $finish;
  end
endmodule
"""

CONFIG = """
[design]
netlist = "cnt2.v"
sdf = "cnt2.sdf"
top = "cnt2"
clock_period_ns = 0.5
[output]
dir = "out"
[library]
cell_functions = ["cells.v"]
[workload]
testbench = ["tb.sv"]
tb_top = "tb"
clock = "clk"
reset = "RESET"
cell_sim_models = ["cells.v"]
n_cycles = 20
padding_cycles = 2
[fault_model]
pulse_widths_ps = [20, 60, 100]
[inject]
jobs = 2
"""


def _project(tmp_path, reset="rst_n", reset_init="0"):
    (tmp_path / "cells.v").write_text(CELLS)
    (tmp_path / "cnt2.v").write_text(NETLIST)
    (tmp_path / "cnt2.sdf").write_text(sdf())
    (tmp_path / "tb.sv").write_text(TB.replace("RESET_INIT", reset_init))
    (tmp_path / "setfi.toml").write_text(CONFIG.replace('"RESET"', f'"{reset}"'))
    return str(tmp_path / "setfi.toml")


def test_run_rerun_and_partial_update(tmp_path, capsys):
    cfg = _project(tmp_path)
    assert cli.main(["run", "-c", cfg]) == 0
    summary = json.loads((tmp_path / "out" / "analyze" / "uniform" / "summary.json").read_text())
    assert summary["n_sites"] == 2 and summary["n_observed_ffs"] == 2
    assert 0 < summary["upset_probability"] < 1
    capsys.readouterr()

    assert cli.main(["run", "-c", cfg]) == 0
    out = capsys.readouterr().out
    assert "[build] up to date" in out and "[record] up to date" in out and "[inject] up to date" in out

    # a comment in the testbench: record runs again, its result is the same
    (tmp_path / "tb.sv").write_text((tmp_path / "tb.sv").read_text() + "// comment\n")
    assert cli.main(["run", "-c", cfg]) == 0
    out = capsys.readouterr().out
    assert "[record] up to date" not in out and "[inject] up to date" in out

    # slower gates: analyze warns until the campaign is redone, record is kept
    (tmp_path / "cnt2.sdf").write_text(sdf("0.045"))
    assert cli.main(["analyze", "-c", cfg]) == 0
    assert "design.sdf" in capsys.readouterr().out
    assert cli.main(["run", "-c", cfg]) == 0
    out = capsys.readouterr().out
    assert "[record] up to date" in out and "[build] up to date" not in out
    # inject re-ran; every site of this counter drives a flip-flop directly, so the
    # records did not change and the analyses were kept
    assert "[inject] up to date" not in out and "removed" not in out
    assert cli.main(["analyze", "-c", cfg]) == 0
    assert "note" not in capsys.readouterr().out


def test_several_width_models(tmp_path):
    cfg = _project(tmp_path)
    with open(cfg, "a") as f:
        f.write('[analyze]\nwidth_models = [{type = "uniform"}, {type = "exponential", tau_ps = 40}]\n')
    assert cli.main(["run", "-c", cfg]) == 0
    names = sorted(os.listdir(tmp_path / "out" / "analyze"))
    assert names == ["exponential_tau40", "uniform"]
    assert cli.main(["analyze", "-c", cfg, "--width-model", '{"type": "gaussian", "mean_ps": 60, "sd_ps": 20}',
                     "--name", "g60"]) == 0
    assert (tmp_path / "out" / "analyze" / "g60" / "summary.json").exists()


def test_design_without_reset(tmp_path):
    # workload.reset empty: cycles are counted from the first clock edge (the
    # testbench still initialises the counter through its reset pin)
    cfg = _project(tmp_path, reset="", reset_init="0")
    assert cli.main(["run", "-c", cfg]) == 0
    assert (tmp_path / "out" / "analyze" / "uniform" / "summary.json").exists()


def test_status_and_a_lost_stages_file(tmp_path, capsys):
    cfg = _project(tmp_path)
    assert cli.main(["run", "-c", cfg]) == 0
    capsys.readouterr()
    with open(cfg) as f:
        text = f.read()
    with open(cfg, "w") as f:
        f.write(text.replace("n_cycles = 20", "n_cycles = 25"))
    assert cli.main(["status", "-c", cfg]) == 0
    out = capsys.readouterr().out
    assert "record   to run  (changed: workload.n_cycles)" in out
    assert "inject   runs again if the result of record changes" in out
    assert "(deleted if the pulse records change)" in out

    with open(cfg, "w") as f:
        f.write(text)
    os.remove(tmp_path / "out" / "stages.json")
    assert cli.main(["analyze", "-c", cfg]) == 0            # inject/ alone is enough
    assert "no entry for the campaign" in capsys.readouterr().out
    assert cli.main(["run", "-c", cfg]) == 0                # everything runs again ...
    out = capsys.readouterr().out
    assert "[inject] up to date" not in out and "removed" not in out   # ... same records
