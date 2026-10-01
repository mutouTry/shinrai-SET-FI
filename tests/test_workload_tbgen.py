"""Tap loading/resolution, generated recorder SV, strobe-event replay, markers."""
import json
import random
import re

import pytest

from setfi.workload import sampling as S
from setfi.workload import tbgen as T
from setfi.workload.errors import WorkloadError
from setfi.workload.simulate import parse_cycle_markers


# ---------------------------------------------------------------- taps
def _net_index(tmp_path, names):
    p = tmp_path / "net_index.json"
    p.write_text(json.dumps({"idx_to_net": {str(i): n for i, n in enumerate(names)},
                             "net_to_idx": {n: i for i, n in enumerate(names)}}))
    return p


def test_load_taps_order_and_flip_flop_nets(tmp_path):
    ni = _net_index(tmp_path, ["a", "n1", "q_reg/Q", "d1"])
    ff = tmp_path / "ff_index.json"
    # the maps list every sequential cell, observed or not
    ff.write_text(json.dumps({"ffid_to_info": {}, "dnet_to_ffids": {"3": [0]},
                              "qnet_to_ffids": {"2": [0]}}))
    taps, ffn = T.load_taps(ni, ff)
    assert taps == ["a", "n1", "q_reg/Q", "d1"]
    assert ffn == {"q_reg/Q", "d1"}
    taps2, ffn2 = T.load_taps(ni)
    assert taps2 == taps and ffn2 is None


def test_load_taps_rejects_sparse_index(tmp_path):
    p = tmp_path / "ni.json"
    p.write_text(json.dumps({"idx_to_net": {"0": "a", "2": "b"}}))
    with pytest.raises(WorkloadError):
        T.load_taps(p)


def test_verify_xmr_resolves():
    text = "module top(a, q); wire n5; DFF u_r ( .Q(q_reg_Q) ); wire [3:0] bus; endmodule"
    taps = ["a", "u_r/q_reg_Q", "bus[2]", "n5", "missing_net", "{n5, a}[0]", "tb.dut.bus[1]"]
    assert T.verify_xmr_resolves(text, taps) == ["missing_net"]


def test_resolve_taps_kinds():
    taps = ["a[0]", "u_dut/q_reg/Q", "1'b1", "ENCLK", "u_x/ENCLK", "m[15:0][3]",
            "{x[1], y}[0]", "{x[1], y}[1]", "u_w/{p[2], q}[1]", "p[2]", "x[1]", "rd[3:0][2][1]"]
    b, meta = T.resolve_taps(taps, "dut", set())
    got = {x.tap: (x.kind, x.expr) for x in b}
    assert got["a[0]"] == ("xmr", "tb.dut.a[0]")
    assert got["u_dut/q_reg/Q"] == ("xmr", "tb.dut.u_dut.q_reg.Q")
    assert got["1'b1"] == ("literal", "1'b1")
    assert got["ENCLK"] == ("gated_clock", "1'b0")
    assert got["u_x/ENCLK"] == ("xmr", "tb.dut.u_x.ENCLK")      # scoped: recorded normally
    assert got["m[15:0][3]"] == ("resolved", "tb.dut.m[3]")
    assert got["{x[1], y}[0]"] == ("resolved", "tb.dut.y")
    assert got["{x[1], y}[1]"] == ("resolved", "tb.dut.x[1]")
    assert got["u_w/{p[2], q}[1]"] == ("resolved", "tb.dut.u_w.p[2]")
    assert got["rd[3:0][2][1]"] == ("unresolved", "1'b0")
    assert meta["n_unresolved_taps"] == 1
    assert [x.idx for x in b] == list(range(len(taps)))


def test_resolve_taps_concat_with_bus_element_is_unresolved():
    # `y` is a bus elsewhere in the tap universe -> the bit position is ambiguous
    taps = ["{x[1], y}[0]", "y[0]", "y[1]"]
    b, meta = T.resolve_taps(taps, "dut", set())
    assert b[0].kind == "unresolved" and meta["n_unresolved_taps"] == 1


def test_resolve_taps_state_tap_is_fatal():
    taps = ["a", "m[0:3][1]"]                        # ascending slice: not resolved
    with pytest.raises(WorkloadError) as e:
        T.resolve_taps(taps, "dut", {"m[0:3][1]"})
    assert e.value.kind == "unresolvable_state_tap"
    with pytest.raises(WorkloadError) as e:          # no flip-flop list -> possibly state
        T.resolve_taps(taps, "dut", None)
    assert e.value.kind == "unresolvable_state_tap"


# ---------------------------------------------------------------- generated SV
def test_rst_guard_expr():
    assert T.rst_guard_expr("rst_n", "active_low") == "tb.rst_n"
    assert T.rst_guard_expr("rst", "active_high") == "!tb.rst"


def test_emit_wrapper(tmp_path):
    p = tmp_path / "w.sv"
    T.emit_wrapper(p, tb_top_module="my_tb", clk_net="ck", rst_net="rst", rst_polarity="active_high",
                   include_path=tmp_path / "rec.svh", version="t")
    s = p.read_text()
    assert "module setfi_tb_wrapper;" in s and "my_tb tb();" in s
    assert "always @(posedge tb.ck)" in s and "if (!tb.rst)" in s
    assert '$display("[SETFI-CYC] last=%0d", cycle_counter);' in s
    assert f'`include "{tmp_path / "rec.svh"}"' in s


def test_emit_recorder_program(tmp_path):
    b, _ = T.resolve_taps(["a", "u/b", "1'b0"], "dut", {"a": "src", "u/b": "ff_q", "1'b0": "aux"})
    T.emit_recorder_program(tmp_path / "r.svh", tmp_path / "p.sv", bindings=b,
                            sample_cycles=[4, 9], clk_net="clk", rst_guard="!tb.rst", version="t")
    svh = (tmp_path / "r.svh").read_text()
    prog = (tmp_path / "p.sv").read_text()
    assert "reg [2:0] setfi_net_state;" in svh
    assert "setfi_cyc_list[    1] = 9;" in svh
    assert "setfi_net_state[    1] = tb.dut.u.b;" in svh
    assert "setfi_net_state[    2] = 1'b0;  // literal-const: \"1'b0\"" in svh
    assert "program setfi_recorder;" in prog
    assert "@(negedge setfi_tb_wrapper.tb.clk);" in prog
    assert "if (!setfi_tb_wrapper.tb.rst && setfi_tb_wrapper.setfi_cyc_ptr < 2) begin" in prog
    assert '$fwrite(_fd, "%h\\n", setfi_tb_wrapper.setfi_net_state);' in prog


def test_emit_recorder_strobe_chunks(tmp_path):
    n = 150                                        # 3 chunks: 64 + 64 + 22
    taps = [f"n{i}" for i in range(n)]
    b, _ = T.resolve_taps(taps, "dut", {t: "src" for t in taps})
    T.emit_recorder_strobe(tmp_path / "r.svh", bindings=b, sample_cycles=[5, 6, 7],
                           clk_net="clk", rst_guard="!tb.rst", version="t")
    s = (tmp_path / "r.svh").read_text()
    assert "wire setfi_guard = (!tb.rst) && 1'b1;" in s
    assert "wire setfi_t149 = tb.dut.n149;" in s
    assert "wire [63:0] setfi_c0 = {" in s and "wire [21:0] setfi_c2 = {" in s
    assert '"%0d %b %h%h%h", cycle_counter, setfi_guard,' in s
    assert "setfi_c2, setfi_c1, setfi_c0);" in s
    # chunk c0 lists t63 (MSB) first and t0 (LSB) last
    c0 = s[s.index("setfi_c0 = {"):s.index("};", s.index("setfi_c0 = {"))]
    names = re.findall(r"setfi_t(\d+)", c0)
    assert names[0] == "63" and names[-1] == "0" and len(names) == 64


def test_chunked_hex_equals_whole_vector_hex():
    """The strobe recorder prints %h per 64-bit chunk, top chunk first; with the
    Verilog %h digit rules that is the %h of the whole vector."""
    rng = random.Random(11)
    for n in (1, 4, 5, 63, 64, 65, 127, 129, 150, 3787, 12182):
        bits = [rng.choice("0000001111111xz") for _ in range(n)]
        whole = S.format_state_hex(bits)
        chunks = [bits[lo:lo + T.STROBE_CHUNK_BITS] for lo in range(0, n, T.STROBE_CHUNK_BITS)]
        assert "".join(S.format_state_hex(c) for c in reversed(chunks)) == whole


# ---------------------------------------------------------------- strobe replay
def _ev(c, g, h):
    return f"{c} {g} {h}"


def test_replay_matches_program_pointer_logic():
    cands = [3, 5, 8]
    h = lambda k: format(k, "02x")
    ev = "\n".join([
        _ev(3, "x", h(1)),       # guard X: not recorded (program: if(x) is false)
        _ev(3, "1", h(2)),       # recorded
        _ev(8, "1", h(3)),       # counter matches a later candidate: pointer is at 5 -> skipped
        _ev(5, "0", h(4)),       # guard 0 -> skipped
        _ev(5, "1", h(5)),       # recorded
        _ev(8, "1", h(6)),       # recorded
        _ev(8, "1", h(7)),       # pointer exhausted
    ]) + "\n"
    states, log = T.replay_strobe_events(ev, cands, 8)
    assert states == {3: "02", 5: "05", 8: "06"}
    assert len(log) == 3


def test_replay_rejects_malformed():
    with pytest.raises(WorkloadError) as e:
        T.replay_strobe_events("3 1\n", [3], 8)
    assert e.value.kind == "recorder_semantics"
    with pytest.raises(WorkloadError):
        T.replay_strobe_events("3 1 abc\n", [3], 8)   # wrong digit count


# ---------------------------------------------------------------- markers
def test_parse_markers():
    log = "x\n[SETFI-CYC] first=0\nRESULT: PASS\n[SETFI-CYC] last=2400\n"
    assert parse_cycle_markers(log, 0) == {"first": 0, "last": 2400}


@pytest.mark.parametrize("log,kind", [
    ("[SETFI-CYC] last=5\n", "sim_no_first_marker"),
    ("[SETFI-CYC] first=0\n", "sim_aborted_no_last_marker"),
    ("[SETFI-CYC] first=0\n[SETFI-CYC] first=1\n[SETFI-CYC] last=5\n", "marker_parse_ambiguous"),
    ("[SETFI-CYC] first=0\n[SETFI-CYC] last=5\n[SETFI-CYC] last=6\n", "marker_parse_ambiguous"),
])
def test_parse_markers_strict(log, kind):
    with pytest.raises(WorkloadError) as e:
        parse_cycle_markers(log, 0)
    assert e.value.kind == kind
