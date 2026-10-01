"""Testbench adaptation (pure text processing; no simulator)."""
import json

import pytest

from setfi.workload.adapt_tb import adapt_testbench
from setfi.workload.errors import WorkloadError

NETLIST = """\
module sub ( a, y );
  input a;
  output y;
  wire [7:0] mem;
  INVD1 u0 ( .I(a), .ZN(y) );
endmodule

module top ( clk, a, y );
  input clk, a;
  output y;
  wire [15:0] u_rf_mem;
  wire u_ctl_state;
  sub u_sub ( .a(a), .y(y) );
endmodule
"""

RTL = """\
module regfile;
  parameter W = 8;
  logic [W-1:0] mem [0:1];
endmodule
module top_rtl;
  regfile u_rf ();
endmodule
"""

TB = """\
module testbench;
  reg clk, rst_n, a;
  wire y;
  top dut (.clk(clk), .a(a), .y(y));
  initial begin
    $display("%h %h", dut.u_rf.mem[1], dut.u_ctl.state);
    $display("%h", dut.u_sub.mem);
    $display("RESULT: PASS");
  end
endmodule
"""


@pytest.fixture
def files(tmp_path):
    (tmp_path / "net.v").write_text(NETLIST)
    (tmp_path / "cells.v").write_text("module INVD1(I, ZN); input I; output ZN; not(ZN, I); endmodule\n")
    (tmp_path / "tb.sv").write_text(TB)
    (tmp_path / "rtl.sv").write_text(RTL)
    return tmp_path


def _adapt(files, **kw):
    args = dict(tb_files=[files / "tb.sv"], netlist=files / "net.v", cell_models=[files / "cells.v"],
                top_module="top", tb_top_module="testbench", out_dir=files / "out")
    args.update(kw)
    return adapt_testbench(**args)


def test_no_adaptation_uses_tb_verbatim(files):
    r = _adapt(files)
    assert r.tb_files == [(files / "tb.sv").resolve()]
    assert r.extra_sources == [] and not r.netlist_was_stripped
    assert r.net_filelist.read_text().splitlines() == [
        str((files / "net.v").resolve()), str((files / "cells.v").resolve()),
        str((files / "tb.sv").resolve())]
    assert json.loads((files / "out" / "result.json").read_text())["status"] == "ok"


def test_manual_substitution(files):
    r = _adapt(files, hier_substitutions=[{"regex": r"dut\.u_ctl\.state", "replacement": "dut.u_ctl_state"}])
    assert r.tb_files == [files / "out" / "tb_netlist.sv"]
    text = r.tb_files[0].read_text()
    assert "dut.u_ctl_state" in text
    assert r.report["metrics"]["n_tb_substitutions_applied"] == 1


def test_auto_derive_flat_and_keep_hier(files):
    r = _adapt(files, auto_derive_hier_substitutions=True, rtl_dut_files=[files / "rtl.sv"])
    text = r.tb_files[0].read_text()
    # flattened indexed array: MSB-first slice of the flat top-level wire (N=2, W=8)
    assert "dut.u_rf_mem[(1-(1))*8 +: 8]" in text
    # flattened scalar
    assert "dut.u_ctl_state" in text
    # keep-hierarchy: resolves natively, kept as is
    assert "dut.u_sub.mem" in text
    kinds = sorted(x["_kind"] for x in r.report["metrics"]["auto_derived_substitutions"])
    assert kinds == ["indexed_msb_first_slice", "keep_hier_noop", "scalar_rename"]


def test_auto_derive_refuses_to_guess(files):
    (files / "tb.sv").write_text(TB.replace("dut.u_ctl.state", "dut.u_nope.state"))
    with pytest.raises(WorkloadError) as e:
        _adapt(files, auto_derive_hier_substitutions=True, rtl_dut_files=[files / "rtl.sv"])
    assert e.value.kind == "tb_unparseable"
    assert json.loads((files / "out" / "result.json").read_text())["status"] == "error"


def test_strip_modules_and_substitutes(files):
    (files / "sub_rtl.v").write_text("module sub(input a, output y); assign y = ~a; endmodule\n")
    r = _adapt(files, netlist_strip_modules=["sub"], netlist_substitute_files=[files / "sub_rtl.v"])
    assert r.netlist_was_stripped
    assert r.sim_netlist == files / "out" / "mapped_adapted.v"
    stripped = r.sim_netlist.read_text()
    assert "module sub" not in stripped and "module top" in stripped
    assert r.extra_sources == [(files / "sub_rtl.v").resolve()]
    with pytest.raises(WorkloadError):
        _adapt(files, netlist_strip_modules=["not_there"])


def test_uniquified_macro_alias(files):
    (files / "net.v").write_text(NETLIST + """
module sram_DEPTH16_WIDTH8 ( clk, q );
  input clk;
  output [7:0] q;
endmodule
""")
    (files / "sram.v").write_text(
        "module sram #(parameter DEPTH = 4, parameter WIDTH = 4) (input clk, output [WIDTH-1:0] q);\n"
        "  assign q = '0;\nendmodule\n")
    r = _adapt(files, macro_rtl_subs=[{"macro_module": "sram", "rtl_file": str(files / "sram.v")}])
    wrappers = files / "out" / "macro_uniquify_wrappers.sv"
    assert "sram #(.DEPTH(16), .WIDTH(8)) u_inner (.clk(clk), .q(q));" in wrappers.read_text()
    assert "module sram_DEPTH16_WIDTH8" not in r.sim_netlist.read_text()
    assert r.extra_sources == [(files / "sram.v").resolve(), wrappers.resolve()]


def test_multiple_tb_files(files):
    (files / "pkg.sv").write_text("package p; endpackage\n// dut.u_ctl.state\n")
    r = _adapt(files, tb_files=[files / "pkg.sv", files / "tb.sv"],
               hier_substitutions=[{"regex": r"dut\.u_ctl\.state", "replacement": "dut.u_ctl_state"}])
    assert [p.name for p in r.tb_files] == ["tb_netlist_0_pkg.sv", "tb_netlist_1_tb.sv"]
    assert all("dut.u_ctl_state" in p.read_text() for p in r.tb_files)
    assert r.report["metrics"]["n_tb_substitutions_applied"] == 2


def test_missing_inputs(files):
    with pytest.raises(WorkloadError) as e:
        _adapt(files, cell_models=[files / "nope.v"])
    assert e.value.kind == "tech_lib_missing"
    with pytest.raises(WorkloadError) as e:
        _adapt(files, hier_substitutions=[{"regex": "("}])
    assert e.value.kind == "config_invalid"
