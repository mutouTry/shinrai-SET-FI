"""The behavioural standard-cell library: pins, sequential inference, cell functions.

The library is an assign-style Verilog file: each combinational cell computes its
outputs with ``assign`` statements, each state-holding cell stores state in an
``always`` block.  From it the build takes

  * pin directions and sequential-cell recognition (graph stage): a cell is a
    flip-flop iff it has a Q-like output, a D-like input and a clock pin, all found by
    the pin-name conventions in :class:`~setfi.build.spec.LibrarySpec`;
  * the Boolean function of every output (timing stage): ``assign`` right-hand sides
    are parsed into an AST and internal wires are substituted until the expression is
    over input pins only;
  * per-arc semantics: which side-pin values let a transition on the arc's input
    reach its output (edge-sensitive patterns, unmask conditions, truth tables).

A cell used by the netlist whose function cannot be recovered is an error
(:func:`check_used_cells`), not an empty entry that silently stops pulses.
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

from .netlist import split_module_blocks, strip_comments
from .spec import BuildError, LibrarySpec

# cell arcs truth tables are enumerated only up to this many variables.
TRUTH_TABLE_MAX_VARS = 8

_RE_WS = re.compile(r"\s+")


def normalize_space(s: str) -> str:
    return _RE_WS.sub(" ", s.strip())


# ============================================================
# Verilog expression AST
# ============================================================
@dataclass(frozen=True)
class AstConst:
    value: int


@dataclass(frozen=True)
class AstVar:
    name: str


@dataclass(frozen=True)
class AstUnary:
    op: str
    arg: "AstNode"


@dataclass(frozen=True)
class AstBinary:
    op: str
    left: "AstNode"
    right: "AstNode"


@dataclass(frozen=True)
class AstTernary:
    cond: "AstNode"
    if_true: "AstNode"
    if_false: "AstNode"


AstNode = Union[AstConst, AstVar, AstUnary, AstBinary, AstTernary]

_TOKEN_RE = re.compile(
    r"""\s*(\d+'[bB][01xXzZ]+|[A-Za-z_][A-Za-z0-9_\[\]\.]*|\|\||&&|==|!=|[~&|^?:()!])""",
    re.X,
)


def tokenize_verilog_expr(expr: str) -> List[str]:
    expr = normalize_space(expr.rstrip(";"))
    pos, out = 0, []
    while pos < len(expr):
        m = _TOKEN_RE.match(expr, pos)
        if not m:
            raise ValueError(f"Bad token in expr near: {expr[pos:pos + 60]}")
        out.append(m.group(1))
        pos = m.end()
    return out


def parse_verilog_const(tok: str) -> int:
    tok = tok.lower()
    if tok in ("0", "1"):
        return int(tok)
    m = re.fullmatch(r"\d+'[bB]([01xXzZ]+)", tok)
    if not m:
        raise ValueError(f"Unsupported const: {tok}")
    bits = m.group(1).lower()
    if any(ch in bits for ch in "xz"):
        return 0
    return int(bits[-1]) & 1


class VerilogExprParser:
    """Recursive-descent parser: ternary > (| ||) > (& &&) > ^ > unary (~ !) > primary."""

    def __init__(self, tokens: List[str]):
        self.toks = tokens
        self.i = 0

    def peek(self) -> Optional[str]:
        return None if self.i >= len(self.toks) else self.toks[self.i]

    def pop(self) -> str:
        if self.i >= len(self.toks):
            raise ValueError("Unexpected EOF")
        t = self.toks[self.i]
        self.i += 1
        return t

    def expect(self, s: str) -> None:
        t = self.pop()
        if t != s:
            raise ValueError(f"Expected {s}, got {t}")

    def parse(self) -> AstNode:
        node = self.parse_ternary()
        if self.peek() is not None:
            raise ValueError(f"Trailing tokens: {self.toks[self.i:]}")
        return node

    def parse_ternary(self) -> AstNode:
        cond = self.parse_or()
        if self.peek() == "?":
            self.pop()
            a = self.parse_ternary()
            self.expect(":")
            b = self.parse_ternary()
            return AstTernary(cond, a, b)
        return cond

    def parse_or(self) -> AstNode:
        node = self.parse_and()
        while self.peek() in ("||", "|"):
            node = AstBinary(self.pop(), node, self.parse_and())
        return node

    def parse_and(self) -> AstNode:
        node = self.parse_xor()
        while self.peek() in ("&&", "&"):
            node = AstBinary(self.pop(), node, self.parse_xor())
        return node

    def parse_xor(self) -> AstNode:
        node = self.parse_unary()
        while self.peek() == "^":
            node = AstBinary(self.pop(), node, self.parse_unary())
        return node

    def parse_unary(self) -> AstNode:
        if self.peek() in ("~", "!"):
            return AstUnary(self.pop(), self.parse_unary())
        return self.parse_primary()

    def parse_primary(self) -> AstNode:
        t = self.peek()
        if t is None:
            raise ValueError("Unexpected EOF in primary")
        if t == "(":
            self.pop()
            node = self.parse_ternary()
            self.expect(")")
            return node
        self.pop()
        if re.fullmatch(r"\d+'[bB][01xXzZ]+|0|1", t):
            return AstConst(parse_verilog_const(t))
        return AstVar(t)


def parse_assign_expr_to_ast(expr: str) -> AstNode:
    return VerilogExprParser(tokenize_verilog_expr(expr)).parse()


def collect_ast_vars(node: AstNode, out: Optional[Set[str]] = None) -> Set[str]:
    if out is None:
        out = set()
    if isinstance(node, AstVar):
        out.add(node.name)
    elif isinstance(node, AstUnary):
        collect_ast_vars(node.arg, out)
    elif isinstance(node, AstBinary):
        collect_ast_vars(node.left, out)
        collect_ast_vars(node.right, out)
    elif isinstance(node, AstTernary):
        collect_ast_vars(node.cond, out)
        collect_ast_vars(node.if_true, out)
        collect_ast_vars(node.if_false, out)
    return out


def substitute_ast(node: AstNode, repl: Dict[str, AstNode]) -> AstNode:
    if isinstance(node, AstConst):
        return node
    if isinstance(node, AstVar):
        return repl.get(node.name, node)
    if isinstance(node, AstUnary):
        return AstUnary(node.op, substitute_ast(node.arg, repl))
    if isinstance(node, AstBinary):
        return AstBinary(node.op, substitute_ast(node.left, repl), substitute_ast(node.right, repl))
    if isinstance(node, AstTernary):
        return AstTernary(
            substitute_ast(node.cond, repl),
            substitute_ast(node.if_true, repl),
            substitute_ast(node.if_false, repl),
        )
    raise TypeError(node)


def eval_ast(node: AstNode, env: Dict[str, int]) -> int:
    if isinstance(node, AstConst):
        return node.value & 1
    if isinstance(node, AstVar):
        return int(env.get(node.name, 0)) & 1
    if isinstance(node, AstUnary):
        a = eval_ast(node.arg, env)
        if node.op in ("~", "!"):
            return 0 if a else 1
        raise ValueError(f"Bad unary op: {node.op}")
    if isinstance(node, AstBinary):
        lv = eval_ast(node.left, env)
        rv = eval_ast(node.right, env)
        if node.op in ("&", "&&"):
            return lv & rv
        if node.op in ("|", "||"):
            return lv | rv
        if node.op == "^":
            return lv ^ rv
        raise ValueError(f"Bad binary op: {node.op}")
    if isinstance(node, AstTernary):
        return eval_ast(node.if_true if eval_ast(node.cond, env) else node.if_false, env)
    raise TypeError(node)


def ast_to_py(node: AstNode) -> str:
    """The AST as a Python expression over 0/1 ints (stored in cell arcs as ``dst_assign_py``)."""
    if isinstance(node, AstConst):
        return str(node.value & 1)
    if isinstance(node, AstVar):
        return node.name
    if isinstance(node, AstUnary):
        if node.op in ("~", "!"):
            return f"(~{ast_to_py(node.arg)} & 1)"
        raise ValueError(node.op)
    if isinstance(node, AstBinary):
        op_map = {"&": "&", "&&": "&", "|": "|", "||": "|", "^": "^"}
        return f"({ast_to_py(node.left)} {op_map[node.op]} {ast_to_py(node.right)})"
    if isinstance(node, AstTernary):
        return (f"({ast_to_py(node.if_true)} if {ast_to_py(node.cond)} "
                f"else {ast_to_py(node.if_false)})")
    raise TypeError(node)


# ============================================================
# Library parsing
# ============================================================
@dataclass
class Cell:
    name: str
    inputs: List[str]           # sorted
    outputs: List[str]          # sorted
    inouts: List[str]           # sorted
    assigns: Dict[str, str]     # lhs -> rhs (whitespace-normalised)
    always_blocks: List[str]
    body: str
    ansi_header: bool = False
    # sequential inference (see LibrarySpec)
    is_sequential: bool = False
    clock_pin: Optional[str] = None
    d_pins: Tuple[str, ...] = ()
    q_pins: Tuple[str, ...] = ()

    @property
    def has_always(self) -> bool:
        return bool(self.always_blocks)


_DECL_KW = {"wire", "reg", "logic", "signed", "unsigned", "tri", "supply0", "supply1",
            "wand", "wor", "uwire", "tri0", "tri1", "triand", "trior", "trireg"}


def _decl_names(body: str, direction: str) -> Set[str]:
    """Pin names declared ``input``/``output``/``inout`` in a cell body.  Ranges and
    net-type keywords are dropped; each comma-separated item names its last word."""
    out: Set[str] = set()
    for decl in re.finditer(r"\b" + direction + r"\b([^;]*);", body):
        items = re.sub(r"\[[^\]]+\]", " ", decl.group(1))
        for item in items.split(","):
            words = [w for w in item.split() if w not in _DECL_KW]
            if words and re.fullmatch(r"\w+", words[-1]):
                out.add(words[-1])
    return out


def _infer_q_pins(outs: Set[str], prefer: Sequence[str]) -> List[str]:
    upper = {p.upper(): p for p in sorted(outs)}
    picks: List[str] = []
    for p in prefer:
        q = upper.get(p.upper())
        if q is not None and q not in picks:
            picks.append(q)
    return picks if picks else sorted(outs)


def _infer_clock_pin(body: str, ins: Set[str], names: Sequence[str]) -> Optional[str]:
    # 1) Behavioral cells with explicit always @(posedge/negedge ...)
    am = re.search(r"always\s*@\s*\(\s*(posedge|negedge)\s+(\w+)", body)
    if am and am.group(2) in ins:
        return am.group(2)
    # 2) Liberty-style specify arcs: (posedge CP => (Q+:D))
    for pat in (
        r"\(\s*(?:posedge|negedge)\s+(\w+)\s*=>\s*\(\s*Q\+\s*:",
        r"\$setuphold\s*\(\s*(?:posedge|negedge)\s+(\w+)",
        r"\$recovery\s*\(\s*(?:posedge|negedge)\s+(\w+)",
        r"\$hold\s*\(\s*(?:posedge|negedge)\s+(\w+)",
    ):
        m = re.search(pat, body)
        if m and m.group(1) in ins:
            return m.group(1)
    # 3) Pin-name convention.
    for p in names:
        if p in ins:
            return p
    return None


def _infer_d_pins(ins: Set[str], clk_pin: Optional[str], d_names: Sequence[str],
                  async_names: Sequence[str]) -> List[str]:
    for p in d_names:
        if p in ins:
            return [p]
    async_like = {a.upper() for a in async_names}
    cands = [p for p in ins if p != clk_pin and p.upper() not in async_like]
    return [sorted(cands)[0]] if cands else []


_ANSI_HEADER_RE = re.compile(r"\bmodule\s+(\w+)\s*\((.*?)\)\s*;", re.S)
_ANSI_DIR_WORDS = re.compile(r"\b(input|output|inout)\b")


def _deansi(text: str) -> str:
    """Rewrite ANSI module headers (``module m (input a, output reg y);``) into the
    non-ANSI form this reader parses: a plain port list plus body declarations."""
    def fix(m: "re.Match[str]") -> str:
        ports_text = m.group(2)
        if not _ANSI_DIR_WORDS.search(ports_text):
            return m.group(0)
        names: List[str] = []
        decls: List[str] = []
        current = ""
        for item in (x.strip() for x in ports_text.split(",")):
            if not item:
                continue
            d = _ANSI_DIR_WORDS.match(item)
            if d:
                current = item[:item.rfind(item.split()[-1])].strip()   # direction + type + range
                name = item.split()[-1]
            else:
                name = item
            names.append(name)
            decls.append(f"  {current} {name};")
        return f"module {m.group(1)} ({', '.join(names)});\n" + "\n".join(decls)
    return _ANSI_HEADER_RE.sub(fix, text)


_ALWAYS_NO_BEGIN_RE = re.compile(
    r"(\balways\s*@\s*(?:\*|\([^)]*\)))(?!\s*begin\b)\s*(.*?)"
    r"(?=\balways\b|\bassign\b|\bendmodule\b|\binitial\b|\bspecify\b|$)", re.S)


def _wrap_always(text: str) -> str:
    """Give every ``always`` without ``begin ... end`` an explicit block."""
    return _ALWAYS_NO_BEGIN_RE.sub(lambda m: f"{m.group(1)} begin {m.group(2).strip()} end\n", text)


def parse_library(text: str, spec: LibrarySpec) -> Dict[str, Cell]:
    """Parse the behavioural library (raw text) into ``{cell name: Cell}``.

    Every module is a cell, including port-less physical cells (fillers, decaps).
    ANSI headers and ``always`` blocks without ``begin/end`` are normalised first.
    """
    text = _wrap_always(_deansi(strip_comments(text)))
    cells: Dict[str, Cell] = {}
    for name, body, ansi in split_module_blocks(text):
        ins = _decl_names(body, "input")
        outs = _decl_names(body, "output")
        inouts = _decl_names(body, "inout")
        assigns: Dict[str, str] = {}
        for am in re.finditer(r"\bassign\s+([^=;]+?)\s*=\s*(.*?);", body, re.S):
            assigns[normalize_space(am.group(1))] = normalize_space(am.group(2))
        always_blocks = [normalize_space(x.group(0))
                         for x in re.finditer(r"\balways\b.*?\bend\b", body, re.S)]
        clk = _infer_clock_pin(body, ins, spec.clock_pin_names)
        q_pins = _infer_q_pins(outs, spec.q_pin_names)
        d_pins = (_infer_d_pins(ins, clk, spec.d_pin_names, spec.async_pin_names)
                  if q_pins else [])
        cells[name] = Cell(
            name=name, inputs=sorted(ins), outputs=sorted(outs), inouts=sorted(inouts),
            assigns=assigns, always_blocks=always_blocks, body=body, ansi_header=ansi,
            is_sequential=bool(q_pins and d_pins and clk), clock_pin=clk,
            d_pins=tuple(d_pins), q_pins=tuple(q_pins),
        )
    if not cells:
        raise BuildError("library.cell_functions defines no cell modules")
    return cells


def is_structurally_untimed(cell: Cell) -> bool:
    """A cell with no input pin or no output pin has no timing arc (tie cells, bus
    holders, fillers, decaps, antenna diodes); SDF writers emit no block for it."""
    return not cell.inputs or not cell.outputs


# ============================================================
# Cell functions
# ============================================================
def _parse_assigns(cell: Cell) -> Tuple[Dict[str, AstNode], Dict[str, str]]:
    parsed: Dict[str, AstNode] = {}
    errors: Dict[str, str] = {}
    for lhs, rhs in cell.assigns.items():
        try:
            parsed[lhs] = parse_assign_expr_to_ast(rhs)
        except Exception as exc:   # noqa: BLE001 -- recorded, surfaced by check_used_cells
            errors[lhs] = f"{rhs!r}: {exc}"
    return parsed, errors


def resolved_output_asts(cell: Cell) -> Dict[str, AstNode]:
    """Output pin -> function AST, internal wires substituted by their own assigns."""
    parsed_assigns, _ = _parse_assigns(cell)

    @lru_cache(maxsize=None)
    def resolve(name: str) -> Optional[AstNode]:
        if name in cell.inputs:
            return AstVar(name)
        if name not in parsed_assigns:
            return None
        node = parsed_assigns[name]
        repl: Dict[str, AstNode] = {}
        for dep in collect_ast_vars(node):
            if dep == name:
                continue
            sub = resolve(dep)
            if sub is not None:
                repl[dep] = sub
        return substitute_ast(node, repl)

    out: Dict[str, AstNode] = {}
    for op in cell.outputs:
        r = resolve(op)
        if r is not None:
            out[op] = r
    return out


def _resolve_assign_ast(name: str, parsed_assigns: Dict[str, AstNode], inputs: Set[str],
                        memo: Dict[str, Optional[AstNode]]) -> Optional[AstNode]:
    if name in memo:
        return memo[name]
    if name in inputs:
        memo[name] = AstVar(name)
        return memo[name]
    if name not in parsed_assigns:
        memo[name] = None
        return None
    node = parsed_assigns[name]
    repl: Dict[str, AstNode] = {}
    for dep in collect_ast_vars(node):
        if dep == name:
            continue
        sub = _resolve_assign_ast(dep, parsed_assigns, inputs, memo)
        if sub is not None:
            repl[dep] = sub
    memo[name] = substitute_ast(node, repl)
    return memo[name]


def check_used_cells(cells: Dict[str, Cell], used: Set[str]) -> List[str]:
    """Problems with the library cells the netlist instantiates (empty list = OK).

    * a cell with state (an ``always`` block) that the pin conventions do not
      recognise as sequential would be treated as combinational logic -- its
      clock/enable pin must be added to ``library.clock_pin_names``;
    * a cell recognised as sequential but with no ``always`` block is a
      combinational cell whose pin names look like a flip-flop's;
    * a combinational cell output whose function cannot be parsed, or depends on
      anything but the cell's inputs, has no truth table: no pulse could be
      propagated through it;
    * a combinational function over more than TRUTH_TABLE_MAX_VARS inputs cannot be
      tabulated;
    * a cell written with an ANSI-style header has no pins as far as this parser is
      concerned.
    """
    problems: List[str] = []
    for name in sorted(used):
        cell = cells.get(name)
        if cell is None:
            continue
        if cell.ansi_header:
            problems.append(f"{name}: port directions are declared in the module header "
                            f"(ANSI style); only body declarations are read")
            continue
        if cell.is_sequential and not cell.has_always:
            problems.append(f"{name}: recognised as sequential (clock {cell.clock_pin}, "
                            f"D {list(cell.d_pins)}, Q {list(cell.q_pins)}) but its model has "
                            f"no `always @(...)` block with a readable event list (see "
                            f"docs/CELLS.md)")
            continue
        if cell.is_sequential:
            continue
        if cell.has_always:
            problems.append(f"{name}: its model holds state (always block) but no clock "
                            f"pin was recognised among inputs {cell.inputs}; add the "
                            f"clock/enable pin to library.clock_pin_names")
            continue
        _, errors = _parse_assigns(cell)
        asts = resolved_output_asts(cell)
        for op in cell.outputs:
            ast = asts.get(op)
            if ast is None:
                why = f"; unparsable assign {errors[op]}" if op in errors else ""
                problems.append(f"{name}: output {op} has no parsable function{why}")
                continue
            vs = collect_ast_vars(ast)
            free = sorted(vs - set(cell.inputs))
            if free:
                detail = "; ".join(f"{k}: {v}" for k, v in sorted(errors.items()))
                problems.append(f"{name}: output {op} depends on {free}, which are not "
                                f"input pins" + (f" (unparsable: {detail})" if detail else ""))
            elif len(vs) > TRUTH_TABLE_MAX_VARS:
                problems.append(f"{name}: output {op} is a function of {len(vs)} inputs; "
                                f"truth tables are limited to {TRUTH_TABLE_MAX_VARS}")
    return problems


# ============================================================
# Arc semantics
# ============================================================
def sort_conjs(conjs) -> List[Dict[str, int]]:
    uniq: Dict[Tuple[Tuple[str, int], ...], Dict[str, int]] = {}
    for c in conjs:
        uniq[tuple(sorted(c.items()))] = dict(c)
    return [uniq[k] for k in sorted(uniq.keys())]


@dataclass
class ArcSemantics:
    kind: str
    dst_expr_py: Optional[str]
    side_pins: List[str]
    event_patterns: Dict[str, Dict[str, List[Dict[str, int]]]]
    unmask: List[Dict[str, int]]
    clock_pin: Optional[str] = None
    clock_edge: Optional[str] = None
    async_pin: Optional[str] = None
    async_edge: Optional[str] = None
    async_value: Optional[int] = None


def _enumerate_patterns(ast: AstNode, src_pin: str, side_pins: List[str]):
    """For every side-pin assignment: does a 0->1 / 1->0 step on src_pin change the
    output, and in which direction.  Returns (event_patterns, unmask) or None."""
    event_patterns: Dict[str, Dict[str, List[Dict[str, int]]]] = {
        "posedge": {"rise": [], "fall": []},
        "negedge": {"rise": [], "fall": []},
    }
    unmask: List[Dict[str, int]] = []
    for bits in itertools.product([0, 1], repeat=len(side_pins)):
        conj = {p: b for p, b in zip(side_pins, bits)}
        any_change = False
        for edge, oldv, newv in (("posedge", 0, 1), ("negedge", 1, 0)):
            env0 = dict(conj)
            env1 = dict(conj)
            env0[src_pin] = oldv
            env1[src_pin] = newv
            try:
                y0 = eval_ast(ast, env0)
                y1 = eval_ast(ast, env1)
            except Exception:   # noqa: BLE001
                return None
            if y0 == y1:
                continue
            any_change = True
            tr = "rise" if (y0 == 0 and y1 == 1) else "fall"
            event_patterns[edge][tr].append(conj)
        if any_change:
            unmask.append(conj)
    return event_patterns, unmask


def _invert_edge(edge: str) -> str:
    return "negedge" if edge == "posedge" else "posedge"


def _find_output_reg(cell: Cell) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for op in cell.outputs:
        rhs = cell.assigns.get(op)
        if rhs and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", rhs):
            out[op] = rhs
    return out


def _extract_always_event_list(always_text: str) -> List[Tuple[str, str]]:
    m = re.search(r"always\s*@\s*\((.*?)\)\s*begin", always_text, re.S)
    if not m:
        return []
    parts = [x.strip() for x in re.split(r"\bor\b", normalize_space(m.group(1))) if x.strip()]
    out: List[Tuple[str, str]] = []
    for p in parts:
        mm = re.match(r"(posedge|negedge)\s+([A-Za-z_][A-Za-z0-9_]*)$", p)
        if mm:
            out.append((mm.group(1), mm.group(2)))
    return out


def _map_internal_event_to_external(pin: str, edge: str, parsed_assigns: Dict[str, AstNode],
                                    inputs: Set[str]) -> Tuple[str, str]:
    if pin in inputs:
        return pin, edge
    ast = parsed_assigns.get(pin)
    if (isinstance(ast, AstUnary) and ast.op in ("~", "!")
            and isinstance(ast.arg, AstVar) and ast.arg.name in inputs):
        return ast.arg.name, _invert_edge(edge)
    return pin, edge


def _extract_seq_assignment_rhs(always_text: str, state_reg: str) -> Optional[str]:
    pats = [
        rf"\belse\b\s+begin\s+.*?\b{re.escape(state_reg)}\s*<=\s*(.*?);",
        rf"\belse\b\s+{re.escape(state_reg)}\s*<=\s*(.*?);",
        rf"\b{re.escape(state_reg)}\s*<=\s*(.*?);",
    ]
    for pat in pats:
        m = re.search(pat, always_text, re.S)
        if m:
            return normalize_space(m.group(1))
    return None


def _extract_async_branch(always_text: str, state_reg: str) -> Optional[Tuple[str, int]]:
    m = re.search(
        rf"\bif\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*(?:begin\s+)?{re.escape(state_reg)}"
        rf"\s*<=\s*(1'b[01]|[01])\s*;",
        always_text, re.S,
    )
    if not m:
        return None
    return m.group(1), parse_verilog_const(m.group(2))


class CellFunctions:
    """Arc semantics of library cells, memoised per (cell type, src pin, dst pin).

    Semantics are a pure function of the cell model, so every instance of a cell
    type shares one result."""

    def __init__(self, cells: Dict[str, Cell], max_enum_side_pins: int,
                 scan_enable_pins: Sequence[str]):
        self.cells = cells
        self.max_enum_side_pins = max_enum_side_pins
        self.scan_enable_pins = tuple(scan_enable_pins)
        self._asts: Dict[str, Dict[str, AstNode]] = {}
        self._memo: Dict[Tuple[str, str, str], Optional[ArcSemantics]] = {}

    def output_asts(self, celltype: str) -> Dict[str, AstNode]:
        r = self._asts.get(celltype)
        if r is None:
            r = resolved_output_asts(self.cells[celltype])
            self._asts[celltype] = r
        return r

    def arc(self, celltype: str, src_pin: str, dst_pin: str) -> Optional[ArcSemantics]:
        key = (celltype, src_pin, dst_pin)
        if key in self._memo:
            return self._memo[key]
        cell = self.cells.get(celltype)
        sem = None
        if cell is not None:
            sem = (self._comb(cell, src_pin, dst_pin)
                   or self._seq(cell, src_pin, dst_pin))
        self._memo[key] = sem
        return sem

    def _comb(self, cell: Cell, src_pin: str, dst_pin: str) -> Optional[ArcSemantics]:
        ast = self.output_asts(cell.name).get(dst_pin)
        if ast is None or src_pin not in cell.inputs:
            return None
        ids = sorted(x for x in collect_ast_vars(ast) if x in cell.inputs)
        if src_pin not in ids:
            return None
        side_pins = [x for x in ids if x != src_pin]
        if len(side_pins) > self.max_enum_side_pins:
            return None
        res = _enumerate_patterns(ast, src_pin, side_pins)
        if res is None:
            return None
        event_patterns, unmask = res
        return ArcSemantics(
            kind="comb", dst_expr_py=ast_to_py(ast), side_pins=side_pins,
            event_patterns={e: {t: sort_conjs(v) for t, v in mp.items()}
                            for e, mp in event_patterns.items()},
            unmask=sort_conjs(unmask),
        )

    def _seq(self, cell: Cell, src_pin: str, dst_pin: str) -> Optional[ArcSemantics]:
        if not cell.has_always or src_pin not in cell.inputs or dst_pin not in cell.outputs:
            return None
        state_reg = _find_output_reg(cell).get(dst_pin)
        if not state_reg:
            return None
        parsed_assigns, _ = _parse_assigns(cell)
        memo: Dict[str, Optional[AstNode]] = {}
        input_set = set(cell.inputs)
        for ab in cell.always_blocks:
            evs = _extract_always_event_list(ab)
            if not evs:
                continue
            rhs_txt = _extract_seq_assignment_rhs(ab, state_reg)
            async_branch = _extract_async_branch(ab, state_reg)
            rhs_ast: Optional[AstNode] = None
            if rhs_txt is not None:
                try:
                    rhs0 = parse_assign_expr_to_ast(rhs_txt)
                    repl: Dict[str, AstNode] = {}
                    for dep in collect_ast_vars(rhs0):
                        sub = _resolve_assign_ast(dep, parsed_assigns, input_set, memo)
                        if sub is not None:
                            repl[dep] = sub
                    rhs_ast = substitute_ast(rhs0, repl)
                except Exception:   # noqa: BLE001
                    rhs_ast = None
            clk_pin = clk_edge = async_pin = async_edge = None
            async_value = None
            if len(evs) >= 1:
                clk_edge0, clk_sig0 = evs[0]
                clk_pin, clk_edge = _map_internal_event_to_external(
                    clk_sig0, clk_edge0, parsed_assigns, input_set)
            if len(evs) >= 2:
                aedge0, asig0 = evs[1]
                async_pin, async_edge = _map_internal_event_to_external(
                    asig0, aedge0, parsed_assigns, input_set)
                if async_branch is not None:
                    async_sig, async_value = async_branch
                    async_pin, async_edge = _map_internal_event_to_external(
                        async_sig, "posedge", parsed_assigns, input_set)
            if rhs_ast is not None:
                ids = sorted(x for x in collect_ast_vars(rhs_ast) if x in input_set)
                if src_pin in ids:
                    side_pins = [x for x in ids if x != src_pin]
                    if len(side_pins) <= self.max_enum_side_pins:
                        res = _enumerate_patterns(rhs_ast, src_pin, side_pins)
                        if res is None:
                            return None
                        event_patterns, unmask = res
                        if unmask:
                            kind = "seq_data"
                            if src_pin == clk_pin:
                                kind = "seq_clock"
                            elif src_pin == async_pin:
                                kind = "seq_async"
                            elif src_pin in self.scan_enable_pins:
                                kind = "seq_scan_enable"
                            return ArcSemantics(
                                kind=kind, dst_expr_py=ast_to_py(rhs_ast),
                                side_pins=side_pins,
                                event_patterns={e: {t: sort_conjs(v) for t, v in mp.items()}
                                                for e, mp in event_patterns.items()},
                                unmask=sort_conjs(unmask),
                                clock_pin=clk_pin, clock_edge=clk_edge,
                                async_pin=async_pin, async_edge=async_edge,
                                async_value=async_value,
                            )
            if src_pin == async_pin:
                return ArcSemantics(
                    kind="seq_async", dst_expr_py=None, side_pins=[],
                    event_patterns={"posedge": {"rise": [], "fall": []},
                                    "negedge": {"rise": [], "fall": []}},
                    unmask=[{}],
                    clock_pin=clk_pin, clock_edge=clk_edge,
                    async_pin=async_pin, async_edge=async_edge, async_value=async_value,
                )
        return None
