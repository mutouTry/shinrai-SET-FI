"""Liberty (.lib) -> cell logic models as assign-style Verilog.

The build stage reads cell functions from Verilog modules written with
``assign`` / ``always`` (see docs/MODEL.md).  Most libraries ship only Liberty
and primitive/UDP-based Verilog, so this module derives the models from the
Liberty ``function``, ``ff`` and ``latch`` descriptions:

    combinational output      assign Z = <function>;
    ff(IQ, IQN)               output reg Q; always @(posedge CLK or <async>) ...
    latch(IQ, IQN)            always @(*) if (<enable>) Q <= <data_in>;
    integrated clock gate     modelled as its latch + output function

Only what the SET model needs is translated.  Cells whose outputs cannot be
expressed (statetable, three-state outputs, buses, complex clocks) are emitted
as a comment naming the reason; the build stage fails if the netlist uses one.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


class LibertyError(ValueError):
    pass


# --------------------------------------------------------------------------- parsing
_COMMENT_RE = re.compile(r"/\*.*?\*/|//[^\n]*", re.S)
_CONT_RE = re.compile(r"\\\s*\n")
_STMT_RE = re.compile(r'\s*([A-Za-z_][\w.]*)\s*')
_BRACE_RE = re.compile(r"[{}]")


@dataclass
class Group:
    kind: str
    args: List[str]
    attrs: Dict[str, str] = field(default_factory=dict)
    groups: List["Group"] = field(default_factory=list)

    def sub(self, kind: str) -> List["Group"]:
        return [g for g in self.groups if g.kind == kind]


def _unquote(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return s[1:-1]
    return s


def _split_args(s: str) -> List[str]:
    return [_unquote(a) for a in s.split(",")] if s.strip() else []


class _Parser:
    KEEP = {"library", "cell", "pin", "bus", "bundle", "ff", "latch", "statetable",
            "ff_bank", "latch_bank", "test_cell"}

    def __init__(self, text: str):
        text = _COMMENT_RE.sub(" ", text)
        self.t = _CONT_RE.sub(" ", text)
        pos = [m.start() for m in _BRACE_RE.finditer(self.t)]
        match: Dict[int, int] = {}
        stack: List[int] = []
        for p in pos:
            if self.t[p] == "{":
                stack.append(p)
            else:
                if not stack:
                    raise LibertyError(f"unbalanced '}}' at offset {p}")
                match[stack.pop()] = p
        if stack:
            raise LibertyError("unbalanced '{'")
        self.match = match

    def _read_until(self, i: int, ch: str) -> int:
        """Index of the next `ch` outside a quoted string, starting at i."""
        t = self.t
        n = len(t)
        in_str = False
        while i < n:
            c = t[i]
            if c == '"':
                in_str = not in_str
            elif c == ch and not in_str:
                return i
            i += 1
        raise LibertyError(f"expected {ch!r}")

    def body(self, start: int, end: int, g: Group) -> None:
        """Parse statements in t[start:end] into g."""
        t = self.t
        i = start
        while True:
            m = _STMT_RE.match(t, i)
            if m is None or m.end() >= end:
                # skip stray separators
                j = i
                while j < end and t[j] in " \t\r\n;":
                    j += 1
                if j >= end:
                    return
                if m is None:
                    raise LibertyError(f"cannot parse Liberty near: {t[j:j + 60]!r}")
            name = m.group(1)
            i = m.end()
            c = t[i] if i < end else ""
            if c == ":":
                j = self._read_until(i + 1, ";")
                g.attrs[name] = _unquote(t[i + 1:j])
                i = j + 1
            elif c == "(":
                j = self._read_until(i + 1, ")")
                args = t[i + 1:j]
                k = j + 1
                while k < end and t[k] in " \t\r\n":
                    k += 1
                if k < end and t[k] == "{":
                    close = self.match[k]
                    if name in self.KEEP:
                        sub = Group(name, _split_args(args))
                        self.body(k + 1, close, sub)
                        g.groups.append(sub)
                    i = close + 1
                else:
                    g.attrs.setdefault(name, args)       # complex attribute
                    i = k + 1 if k < end and t[k] == ";" else k
            elif c == ";":
                i += 1
            else:
                raise LibertyError(f"cannot parse Liberty near: {t[m.start():m.start() + 60]!r}")

    def parse(self) -> Group:
        root = Group("root", [])
        self.body(0, len(self.t), root)
        libs = root.sub("library")
        if not libs:
            raise LibertyError("no library group")
        return libs[0]


def parse_liberty(text: str) -> Group:
    return _Parser(text).parse()


# --------------------------------------------------------------------------- expressions
_TOK_RE = re.compile(r"\s*(?:([A-Za-z_][\w\[\]\.]*)|(\d+)|(\S))")


def _tokens(expr: str) -> List[Tuple[str, str]]:
    out = []
    i = 0
    expr = expr.strip()
    while i < len(expr):
        m = _TOK_RE.match(expr, i)
        if m is None or m.end() == i:
            break
        i = m.end()
        if m.group(1):
            out.append(("id", m.group(1)))
        elif m.group(2):
            out.append(("num", m.group(2)))
        elif m.group(3):
            out.append(("op", m.group(3)))
    return out


class _Expr:
    """Liberty Boolean expression -> fully parenthesised Verilog.

    Precedence (Liberty): inversion (!x, x') > ^ > AND (&, *, juxtaposition) > OR (|, +).
    """

    def __init__(self, expr: str):
        self.toks = _tokens(expr)
        self.i = 0
        self.src = expr

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else ("eof", "")

    def take(self):
        t = self.peek()
        self.i += 1
        return t

    def parse(self) -> str:
        v = self.or_()
        if self.peek()[0] != "eof":
            raise LibertyError(f"trailing tokens in function {self.src!r}")
        return v

    def or_(self):
        a = self.and_()
        while self.peek() in (("op", "|"), ("op", "+")):
            self.take()
            a = f"({a} | {self.and_()})"
        return a

    def _starts_operand(self):
        k, v = self.peek()
        return k in ("id", "num") or (k == "op" and v in ("(", "!"))

    def and_(self):
        a = self.xor_()
        while True:
            if self.peek() in (("op", "&"), ("op", "*")):
                self.take()
            elif not self._starts_operand():
                return a
            a = f"({a} & {self.xor_()})"

    def xor_(self):
        a = self.unary()
        while self.peek() == ("op", "^"):
            self.take()
            a = f"({a} ^ {self.unary()})"
        return a

    def unary(self):
        if self.peek() == ("op", "!"):
            self.take()
            return f"(!{self.unary()})"
        k, v = self.take()
        if k == "id":
            a = v
        elif k == "num":
            if v not in ("0", "1"):
                raise LibertyError(f"constant {v} in function {self.src!r}")
            a = f"1'b{v}"
        elif (k, v) == ("op", "("):
            a = self.or_()
            if self.take() != ("op", ")"):
                raise LibertyError(f"missing ')' in function {self.src!r}")
        else:
            raise LibertyError(f"unexpected {v!r} in function {self.src!r}")
        while self.peek() == ("op", "'"):
            self.take()
            a = f"(!{a})"
        return a


def to_verilog_expr(expr: str) -> str:
    return _Expr(expr).parse()


def expr_vars(expr: str) -> List[str]:
    return sorted({v for k, v in _tokens(expr) if k == "id"})


# --------------------------------------------------------------------------- cells -> Verilog
@dataclass
class CellModel:
    name: str
    verilog: Optional[str]
    kind: str            # comb | ff | latch | icg | tie | physical | unsupported
    reason: str = ""


def _strip_parens(e: str) -> str:
    e = e.replace(" ", "")
    while e.startswith("(") and e.endswith(")") and _balanced(e[1:-1]):
        e = e[1:-1]
    return e


def _balanced(e: str) -> bool:
    d = 0
    for c in e:
        d += (c == "(") - (c == ")")
        if d < 0:
            return False
    return d == 0


def _edge_of(clock_expr: str, inputs: List[str]) -> Tuple[str, str]:
    """clocked_on -> (edge, pin): 'CP' -> posedge CP, '!CPN' / "CPN'" -> negedge CPN."""
    e = _strip_parens(clock_expr)
    if e in inputs:
        return "posedge", e
    m = re.fullmatch(r"!([A-Za-z_]\w*)|([A-Za-z_]\w*)'", e)
    if m:
        pin = m.group(1) or m.group(2)
        if pin in inputs:
            return "negedge", pin
    raise LibertyError(f"clock expression {clock_expr!r} is not a single (inverted) pin")


def _async(expr: Optional[str], inputs: List[str]) -> Optional[Tuple[str, str, str]]:
    """clear/preset -> (edge, pin, verilog condition) for a single (inverted) pin."""
    if not expr:
        return None
    e = _strip_parens(expr)
    if e in inputs:
        return "posedge", e, e
    m = re.fullmatch(r"!([A-Za-z_]\w*)|([A-Za-z_]\w*)'", e)
    if m and (m.group(1) or m.group(2)) in inputs:
        pin = m.group(1) or m.group(2)
        return "negedge", pin, f"!{pin}"
    raise LibertyError(f"asynchronous control {expr!r} is not a single (inverted) pin")


def cell_to_verilog(cell: Group) -> CellModel:
    name = cell.args[0] if cell.args else "?"
    pins = cell.sub("pin")
    if cell.sub("bus") or cell.sub("bundle"):
        return CellModel(name, None, "unsupported", "bus/bundle pins")
    inputs = [p.args[0] for p in pins if p.attrs.get("direction") == "input"]
    outputs = [p for p in pins if p.attrs.get("direction") in ("output", "inout")]
    if not outputs:
        return CellModel(name, None, "physical", "no output pins")
    if cell.attrs.get("clock_gating_integrated_cell"):
        return _icg_to_verilog(name, cell, pins, inputs, outputs)
    if cell.sub("statetable"):
        return CellModel(name, None, "unsupported", "statetable")
    for p in outputs:
        if p.attrs.get("three_state"):
            return CellModel(name, None, "unsupported", "three-state output")
    port_list = ", ".join(inputs + [p.args[0] for p in outputs])
    lines = [f"module {name} ({port_list});"]
    if inputs:
        lines.append(f"    input {', '.join(inputs)};")

    ffs, latches = cell.sub("ff"), cell.sub("latch")
    if len(ffs) + len(latches) > 1:
        return CellModel(name, None, "unsupported", "more than one ff/latch group")
    state = (ffs or latches or [None])[0]
    if state is None:
        for p in outputs:
            f = p.attrs.get("function")
            if f is None:
                return CellModel(name, None, "unsupported", f"output {p.args[0]} has no function")
        lines += [f"    output {', '.join(p.args[0] for p in outputs)};"]
        for p in outputs:
            lines.append(f"    assign {p.args[0]} = {to_verilog_expr(p.attrs['function'])};")
        lines.append("endmodule")
        kind = "tie" if not inputs else "comb"
        return CellModel(name, "\n".join(lines), kind)

    iq = state.args[0] if state.args else "IQ"
    iqn = state.args[1] if len(state.args) > 1 else "IQN"
    # the state output: the output whose function is exactly IQ becomes `output reg`
    q_out = next((p for p in outputs if (p.attrs.get("function") or "").replace(" ", "") == iq), None)
    reg = q_out.args[0] if q_out is not None else iq
    others = [p for p in outputs if p is not q_out]
    if q_out is not None:
        lines.append(f"    output reg {reg};")
    else:
        lines.append(f"    reg {reg};")
    if others:
        lines.append(f"    output {', '.join(p.args[0] for p in others)};")

    def _sub_state(expr: str) -> str:
        v = to_verilog_expr(expr)
        v = re.sub(rf"\b{re.escape(iqn)}\b", f"(!{reg})", v)
        return re.sub(rf"\b{re.escape(iq)}\b", reg, v)

    if ffs:
        edge, clk = _edge_of(state.attrs.get("clocked_on", ""), inputs)
        nxt = state.attrs.get("next_state")
        if not nxt:
            return CellModel(name, None, "unsupported", "ff without next_state")
        clr = _async(state.attrs.get("clear"), inputs)
        pre = _async(state.attrs.get("preset"), inputs)
        sens = [f"{edge} {clk}"] + [f"{a[0]} {a[1]}" for a in (clr, pre) if a]
        lines.append(f"    always @({' or '.join(sens)}) begin")
        branch = "if"
        if clr:
            lines.append(f"        {branch} ({clr[2]}) {reg} <= 1'b0;")
            branch = "else if"
        if pre:
            lines.append(f"        {branch} ({pre[2]}) {reg} <= 1'b1;")
            branch = "else"
        if branch == "if":
            lines.append(f"        {reg} <= {_sub_state(nxt)};")
        else:
            lines.append(f"        else {reg} <= {_sub_state(nxt)};")
        lines.append("    end")
        kind = "ff"
    else:
        en = state.attrs.get("enable")
        din = state.attrs.get("data_in")
        if not en or not din:
            return CellModel(name, None, "unsupported", "latch without enable/data_in")
        lines.append("    always @(*) begin")
        lines.append(f"        if ({to_verilog_expr(en)}) {reg} <= {_sub_state(din)};")
        lines.append("    end")
        kind = "icg" if cell.attrs.get("clock_gating_integrated_cell") else "latch"
    for p in others:
        f = p.attrs.get("function")
        if f is None:
            return CellModel(name, None, "unsupported", f"output {p.args[0]} has no function")
        lines.append(f"    assign {p.args[0]} = {_sub_state(f)};")
    lines.append("endmodule")
    return CellModel(name, "\n".join(lines), kind)


def _icg_to_verilog(name: str, cell: Group, pins: List[Group], inputs: List[str],
                    outputs: List[Group]) -> CellModel:
    """Integrated clock gate from its clock_gating_integrated_cell type and pin roles.

    latch_posedge*: the latch is transparent while the clock is low, GCLK = CLK & latch.
    latch_negedge*: transparent while the clock is high, GCLK = CLK | !latch.
    *_precontrol: the test pin is ORed into the latch input; *_postcontrol: after it.
    """
    kind = str(cell.attrs["clock_gating_integrated_cell"]).strip()
    role = lambda attr: [p.args[0] for p in pins if str(p.attrs.get(attr, "")).strip() == "true"]  # noqa: E731
    clk, en, te, gout = role("clock_gate_clock_pin"), role("clock_gate_enable_pin"), \
        role("clock_gate_test_pin"), role("clock_gate_out_pin")
    if len(clk) != 1 or len(en) != 1 or len(gout) != 1 or len(outputs) != 1 or len(te) > 1:
        return CellModel(name, None, "unsupported", f"clock gate pin roles unclear ({kind})")
    m = re.fullmatch(r"latch_(posedge|negedge)(_precontrol|_postcontrol)?(_obs)?", kind)
    if m is None:
        return CellModel(name, None, "unsupported", f"clock gate type {kind}")
    pos, post = m.group(1) == "posedge", m.group(2) == "_postcontrol"
    c, e, g = clk[0], en[0], gout[0]
    data = e if (post or not te) else f"({e} | {te[0]})"
    held = f"(IQ | {te[0]})" if (post and te) else "IQ"
    port_list = ", ".join(inputs + [g])
    lines = [f"module {name} ({port_list});", f"    input {', '.join(inputs)};", f"    output {g};",
             "    reg IQ;", "    always @(*) begin", f"        if ({'!' if pos else ''}{c}) IQ <= {data};",
             "    end",
             f"    assign {g} = {c} & {held};" if pos else f"    assign {g} = {c} | (!{held});",
             "endmodule"]
    return CellModel(name, "\n".join(lines), "icg")


def liberty_to_verilog(paths: List[str]) -> Tuple[str, Dict[str, CellModel]]:
    """Translate one or more Liberty files; the first definition of a cell wins."""
    models: Dict[str, CellModel] = {}
    for path in paths:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lib = parse_liberty(f.read())
        for cell in lib.sub("cell"):
            try:
                m = cell_to_verilog(cell)
            except LibertyError as e:
                m = CellModel(cell.args[0] if cell.args else "?", None, "unsupported", str(e))
            models.setdefault(m.name, m)
    out = ["// Generated by setfi from Liberty: " + ", ".join(paths), "`timescale 1ns/1ps", ""]
    for m in models.values():
        if m.verilog:
            out += [m.verilog, ""]
        else:
            out += [f"// {m.name}: not modelled ({m.reason})", ""]
    return "\n".join(out), models
