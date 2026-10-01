"""End-to-end record_workload on Icarus Verilog (skipped when not installed).

A one-flip-flop design whose testbench drives its input on the falling edge
(blocking) and releases reset with a non-blocking assignment on a falling edge:
exactly the situation in which an Active-region recorder reads the previous
cycle's stimulus.  Every recorded bit is checked against a Python model.
"""
import json
import random
import shutil

import pytest

from setfi.workload import WorkloadSpec, icarus_preset, record_workload
from setfi.workload.sampling import format_state_hex

pytestmark = pytest.mark.skipif(shutil.which("iverilog") is None or shutil.which("vvp") is None,
                                reason="Icarus Verilog not installed")

CELLS = """\
`timescale 1ns/1ps
module DFFR (D, CP, RN, Q);
  input D, CP, RN; output reg Q;
  always @(posedge CP or negedge RN) if (!RN) Q <= 1'b0; else Q <= D;
endmodule
module XOR2 (A1, A2, Z);
  input A1, A2; output Z;
  xor (Z, A1, A2);
endmodule
"""

NETLIST = """\
module top ( clk, rst_n, a, q );
  input clk, rst_n, a;
  output q;
  wire n1;
  XOR2 u_x ( .A1(a), .A2(q), .Z(n1) );
  DFFR q_reg ( .D(n1), .CP(clk), .RN(rst_n), .Q(q) );
endmodule
"""

N_CYC = 40


def _tb(stim):
    bits = "".join(str(b) for b in stim)
    return f"""\
`timescale 1ns/1ps
module testbench;
  reg clk, rst_n, a;
  wire q;
  reg [0:{N_CYC - 1}] stim;
  integer i;
  top dut (.clk(clk), .rst_n(rst_n), .a(a), .q(q));
  initial clk = 1'b0;
  always #5 clk = ~clk;
  initial begin
    stim = {N_CYC}'b{bits};
    rst_n = 1'b0; a = 1'b0;
    repeat (3) @(negedge clk);
    rst_n <= 1'b1;
    for (i = 0; i < {N_CYC}; i = i + 1) begin
      a = stim[i];
      @(negedge clk);
    end
    $display("RESULT: PASS");
    $finish;
  end
endmodule
"""


def test_record_workload_icarus_end_to_end(tmp_path):
    rng = random.Random(5)
    stim = [rng.randint(0, 1) for _ in range(N_CYC)]
    nets = ["a", "clk", "n1", "q", "rst_n"]
    (tmp_path / "cells.v").write_text(CELLS)
    (tmp_path / "net.v").write_text(NETLIST)
    (tmp_path / "tb.sv").write_text(_tb(stim))
    (tmp_path / "ni.json").write_text(json.dumps({"idx_to_net": {str(i): n for i, n in enumerate(nets)}}))
    (tmp_path / "ff.json").write_text(json.dumps({"ffid_to_info": {"0": {"d_net": "n1"}},
                                                  "dnet_to_ffids": {str(nets.index("n1")): [0]},
                                                  "qnet_to_ffids": {str(nets.index("q")): [0]}}))

    spec = WorkloadSpec(
        netlist=tmp_path / "net.v", top_module="top", net_index=tmp_path / "ni.json",
        ff_index=tmp_path / "ff.json", testbench=[tmp_path / "tb.sv"],
        cell_models=[tmp_path / "cells.v"], output_dir=tmp_path / "out",
        simulator=icarus_preset(), n_sample_cycles=5, sample_seed=3, padding_cycles=2,
        compile_timeout_s=120, sim_timeout_s=120,
    )
    res = record_workload(spec)
    meta = json.loads((tmp_path / "out" / "cycles.json").read_text())
    assert meta["format"] == "setfi-cycles/1"
    assert meta["cycles_after_reset"] == N_CYC
    assert len(meta["sampling"]["candidates"]) == 20       # 5 primary + 15 reserve
    cycles = meta["cycles"]
    assert len(cycles) == 5 and all(2 <= c <= N_CYC - 2 for c in cycles)

    # Model: at the falling edge of cycle N the testbench has applied stim[N]
    # and q holds q_N, where q_0 = 0 and q_{N+1} = stim[N] ^ q_N.
    q = [0]
    for s in stim:
        q.append(s ^ q[-1])
    files = sorted(p.name for p in (tmp_path / "out" / "states").iterdir())
    assert files == sorted(f"cycle_{c}.hex" for c in cycles)
    for c in cycles:
        expect = {"a": stim[c], "clk": 0, "n1": stim[c] ^ q[c], "q": q[c], "rst_n": 1}
        text = (tmp_path / "out" / "states" / f"cycle_{c}.hex").read_text()
        assert text == format_state_hex([expect[n] for n in nets]) + "\n", c
    assert res["status"] == "ok"


def _late_tb(release="rst_n <= 1'b1;", late=""):
    return f"""\
`timescale 1ns/1ps
module testbench;
  reg clk, rst_n, a;
  wire q;
  integer i;
  top dut (.clk(clk), .rst_n(rst_n), .a(a), .q(q));
  initial clk = 1'b0;
  always #5 clk = ~clk;
  initial begin
    rst_n = 1'b0; a = 1'b0;
    repeat (3) @(negedge clk);
    #1 {release}
    for (i = 0; i < {N_CYC}; i = i + 1) begin
      a = i[0];
      {late}
      @(negedge clk);
    end
    $display("RESULT: PASS");
    $finish;
  end
endmodule
"""


@pytest.mark.parametrize("late, error", [
    ("", None),                                              # reset released mid-phase: fine
    ("if (i == 20) begin #2 a = ~a; end", "'a' between a falling"),
    ("#2 a = ~a;", "'a' between a falling"),                 # every cycle
])
def test_input_timing_check(tmp_path, late, error):
    from setfi.workload.errors import WorkloadError
    nets = ["a", "clk", "n1", "q", "rst_n"]
    (tmp_path / "cells.v").write_text(CELLS)
    (tmp_path / "net.v").write_text(NETLIST)
    (tmp_path / "tb.sv").write_text(_late_tb(late=late))
    (tmp_path / "ni.json").write_text(json.dumps({"idx_to_net": {str(i): n for i, n in enumerate(nets)}}))
    spec = WorkloadSpec(
        netlist=tmp_path / "net.v", top_module="top", net_index=tmp_path / "ni.json",
        testbench=[tmp_path / "tb.sv"], cell_models=[tmp_path / "cells.v"],
        output_dir=tmp_path / "out", simulator=icarus_preset(), n_sample_cycles=5,
        padding_cycles=2, compile_timeout_s=120, sim_timeout_s=120)
    if error is None:
        assert record_workload(spec)["status"] == "ok"
    else:
        with pytest.raises(WorkloadError, match=error):
            record_workload(spec)
