import itertools
import re

import pytest

from setfi.library.liberty import (LibertyError, cell_to_verilog, liberty_to_verilog,
                                   parse_liberty, to_verilog_expr)

LIB = r"""
/* a tiny library */
library (tiny) {
  delay_model : table_lookup;
  lu_table_template (t1) { variable_1 : input_net_transition; index_1 ("1, 2"); }
  cell (AOI21X1) {
    area : 1.0;
    pin (A1) { direction : input; capacitance : 0.001; }
    pin (A2) { direction : input; }
    pin (B)  { direction : input; }
    pin (ZN) { direction : output; function : "!((A1 A2) + B)";
      timing () { related_pin : "A1"; cell_rise (t1) { values ("1, 2"); } }
    }
  }
  cell (XOR2X1) {
    pin (A) { direction : input; } pin (B) { direction : input; }
    pin (Z) { direction : output; function : "(A^B)"; }
  }
  cell (MUX2X1) {
    pin (I0) { direction : input; } pin (I1) { direction : input; } pin (S) { direction : input; }
    pin (Z) { direction : output; function : "(I0 S') + (I1 * S)"; }
  }
  cell (DFFRX1) {
    ff (IQ, IQN) { next_state : "D"; clocked_on : "CK"; clear : "(!RN)"; }
    pin (D) { direction : input; } pin (CK) { direction : input; clock : true; }
    pin (RN) { direction : input; }
    pin (Q) { direction : output; function : "IQ"; }
    pin (QN) { direction : output; function : "IQN"; }
  }
  cell (DFFNX1) {
    ff (IQ, IQN) { next_state : "D"; clocked_on : "!CKN"; }
    pin (D) { direction : input; } pin (CKN) { direction : input; }
    pin (Q) { direction : output; function : "IQ"; }
  }
  cell (ICGX1) {
    clock_gating_integrated_cell : latch_posedge_precontrol;
    pin (CK) { direction : input; clock_gate_clock_pin : true; }
    pin (E) { direction : input; clock_gate_enable_pin : true; }
    pin (SE) { direction : input; clock_gate_test_pin : true; }
    pin (GCK) { direction : output; clock_gate_out_pin : true; state_function : "CK * IQ"; }
  }
  cell (TIEHI) { pin (Z) { direction : output; function : "1"; } }
  cell (FILL1) { area : 1; }
  cell (TBUF) {
    pin (A) { direction : input; } pin (OE) { direction : input; }
    pin (Z) { direction : output; function : "A"; three_state : "!OE"; }
  }
}
"""


def _truth(verilog_expr, names):
    py = verilog_expr.replace("!", " not ").replace("&", " and ").replace("|", " or ") \
        .replace("^", " != ").replace("1'b1", "True").replace("1'b0", "False")
    out = []
    for bits in itertools.product([False, True], repeat=len(names)):
        out.append(bool(eval(py, {}, dict(zip(names, bits)))))
    return out


@pytest.mark.parametrize("expr,names,ref", [
    ("!((A1 A2) + B)", ["A1", "A2", "B"], lambda a1, a2, b: not ((a1 and a2) or b)),
    ("A1' * A2 | B", ["A1", "A2", "B"], lambda a1, a2, b: ((not a1) and a2) or b),
    ("A ^ B C", ["A", "B", "C"], lambda a, b, c: (a != b) and c),     # ^ binds tighter than AND
    ("!A B", ["A", "B"], lambda a, b: (not a) and b),
    ("(A+B)'", ["A", "B"], lambda a, b: not (a or b)),
])
def test_expression_precedence(expr, names, ref):
    v = to_verilog_expr(expr)
    assert _truth(v, names) == [ref(*bits) for bits in itertools.product([False, True], repeat=len(names))]


def test_translate_tiny_library():
    text, models = liberty_to_verilog_from_text(LIB)
    kinds = {n: m.kind for n, m in models.items()}
    assert kinds == {"AOI21X1": "comb", "XOR2X1": "comb", "MUX2X1": "comb", "DFFRX1": "ff",
                     "DFFNX1": "ff", "ICGX1": "icg", "TIEHI": "tie", "FILL1": "physical",
                     "TBUF": "unsupported"}
    assert re.search(r"always @\(posedge CK or negedge RN\)", models["DFFRX1"].verilog)
    assert "output reg Q;" in models["DFFRX1"].verilog
    assert "assign QN = (!Q);" in models["DFFRX1"].verilog
    assert "always @(negedge CKN)" in models["DFFNX1"].verilog
    assert "if (!CK) IQ <= (E | SE);" in models["ICGX1"].verilog
    assert "assign GCK = CK & IQ;" in models["ICGX1"].verilog
    assert "assign Z = 1'b1;" in models["TIEHI"].verilog
    assert "TBUF: not modelled (three-state output)" in text


def test_unbalanced_braces():
    with pytest.raises(LibertyError):
        parse_liberty("library (x) { cell (a) { pin (A) { direction : input; }")


def liberty_to_verilog_from_text(text):
    lib = parse_liberty(text)
    models = {}
    for c in lib.sub("cell"):
        m = cell_to_verilog(c)
        models[m.name] = m
    out = "\n".join(m.verilog if m.verilog else f"// {m.name}: not modelled ({m.reason})"
                    for m in models.values())
    return out, models
