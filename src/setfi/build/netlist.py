"""Gate-level netlist parser with hierarchical elaboration.

Given the text of a structural (mapped) Verilog netlist, ``parse_modules`` returns every
``module ... endmodule`` block as a :class:`ModuleDef` (ports, declared ranges, child
instances with their named-port connections).  ``elaborate`` walks the hierarchy from the
top module and returns one :class:`Instance` per leaf, keyed by its hierarchical path
(``/`` separator, the SDF divider), with every pin resolved to a GLOBAL net name:
sub-module ports are replaced by whatever the parent connected, local wires are prefixed
with the instance path.

Hierarchy (wrapper modules that synthesis kept unflattened) is the normal case, not an
option: a parser that only reads the top body misses every leaf inside a wrapper.

Known constructs this parser does NOT model (they are reported, not silently accepted,
where it can tell):
  * ``assign`` statements are modelled only as net aliases (``assign a = b;``, bit-wise,
    with buses, slices, concatenations and constants): the left-hand net is merged into
    the right-hand one.  An ``assign`` with logic on its right-hand side is refused.
  * positional port connections, instance arrays, generate blocks, ANSI-style module
    headers of sub-modules (port lists are read from body declarations).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

from .spec import BuildError


# ============================================================
# Data structures
# ============================================================
@dataclass
class Instance:
    name: str
    celltype: str
    pin2net: Dict[str, str]


@dataclass
class ModuleDef:
    """One ``module ... endmodule`` block from the netlist.

    ports       -- set of port identifiers (input/output/inout names, bus brackets stripped)
    port_dirs   -- {port: 'input'|'output'|'inout'}
    port_ranges -- {port: (msb, lsb)} for the declared vector ports only; scalar
                   ports are absent.  Needed to place a child's port BIT inside a
                   parent-side connection EXPRESSION, which is positional in
                   Verilog and therefore unresolvable without the declared range.
    net_ranges  -- {net: (msb, lsb)} for every declared vector in the module,
                   ports included.  A concatenation element may be a bare vector
                   name whose width is declared only here.
    instances   -- list of child instances declared inside the module body
    n_assign    -- number of `assign` statements in the body
    assigns     -- their (lhs, rhs) source text
    ansi_header -- the port directions are declared in the header (not supported)
    positional  -- names of instances connected by position (not supported)
    """
    name: str
    ports: Set[str] = field(default_factory=set)
    instances: List[Instance] = field(default_factory=list)
    port_ranges: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    net_ranges: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    port_dirs: Dict[str, str] = field(default_factory=dict)
    n_assign: int = 0
    assigns: List[Tuple[str, str]] = field(default_factory=list)
    ansi_header: bool = False
    positional: List[str] = field(default_factory=list)


# ============================================================
# Comments
# ============================================================
# One left-to-right scan: a comment opener inside a string, an escaped identifier or the
# other comment kind is not a comment opener.  (Two sequential regex passes -- line then
# block, or block then line -- each mis-strip one of those shapes.)
_COMMENT_OR_KEEP_RE = re.compile(
    r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\\n])*"|\\\S+', re.S)


def _comment_sub(m: "re.Match[str]") -> str:
    s = m.group(0)
    if s.startswith("//") or s.startswith("/*"):
        return ""
    return s


def strip_comments(text: str) -> str:
    return _COMMENT_OR_KEEP_RE.sub(_comment_sub, text)


# ============================================================
# Per-module structural parser
# ============================================================
# Match a module header up through the first ';' that closes its port list.
_MODULE_HEADER_RE = re.compile(
    r"\bmodule\s+(\w+)\s*(?:\([^)]*\))?\s*;",
    re.S,
)

# Every net/variable type that can carry a declared range in a structural
# netlist.  `tri` is not exotic here: DC writes it for any net it has given more
# than one driver.  A type missing from this list makes its nets look SCALAR, which
# mis-sizes a bare vector element inside a concatenation and mis-places a child's
# port bit -- both silently.  Longest-first so `tri0` is not shadowed by `tri`.
_NET_TYPE_KW = (r"trireg|triand|trior|tri0|tri1|tri|wand|wor|uwire"
                r"|supply0|supply1|wire|reg|logic")

# Port-direction declarations inside a module body.  Tolerates bus widths
# (``[3:0]``), net-type qualifiers, multiline lists.
_PORT_DECL_RE = re.compile(
    r"\b(input|output|inout)\b((?:\s*\[[^\]]+\])?\s*(?:" + _NET_TYPE_KW +
    r")?\s*[^;]*?);",
    re.S,
)

# The leading ``[msb:lsb]`` of a vector port declaration, if present.
_PORT_RANGE_RE = re.compile(r"^\[\s*(-?\d+)\s*:\s*(-?\d+)\s*\]")

# Local vector declarations: ``wire [7:0] a, b;`` / ``reg [3:0] s;``.  Needed
# because a concatenation element may be a bare VECTOR name, and its width is
# declared here and nowhere else.
_NET_DECL_RE = re.compile(
    r"\b(?:" + _NET_TYPE_KW + r")\b\s*\[\s*(-?\d+)\s*:\s*(-?\d+)\s*\]([^;]*);",
    re.S)

# A single-bit reference and a contiguous slice of an already-resolved net name.
# The base is anything WITHOUT brackets, so a hierarchical path (`u_dut/src1_reg`)
# matches while a 2-D reference (`mem[3][2]`) deliberately does not -- the latter
# falls through to the width check below and is REFUSED rather than mis-sliced.
_BITREF_RE = re.compile(r"^([^\[\]]+)\[(-?\d+)\]$")
_SLICE_RE = re.compile(r"^([^\[\]]+)\[\s*(-?\d+)\s*:\s*(-?\d+)\s*\]$")
# A sized Verilog constant, e.g. 4'b0101 / 8'h0f / 1'b0.
_SIZED_CONST_RE = re.compile(r"^(\d+)'[sS]?([bBoOhHdD])([0-9a-fA-FxXzZ_?]+)$")

_ASSIGN_RE = re.compile(r"\bassign\b")
_ASSIGN_STMT_RE = re.compile(r"\bassign\s+([^;]*?)\s*=\s*([^;]*?)\s*;", re.S)


_ANSI_DIR_RE = re.compile(r"\b(?:input|output|inout)\b")


def split_module_blocks(text: str) -> List[Tuple[str, str, bool]]:
    """Walk the (already comment-stripped) text and return
    ``(module_name, body_text, ansi_header)`` for every ``module ... endmodule`` block.

    Each body is the text between the closing ``;`` of the header and the matching
    ``endmodule``.  Headers and ``endmodule``s are paired by sequential scan -- modules
    never nest, so this is unambiguous.  ``ansi_header`` is true when the header's port
    list carries directions (``module m (input a, output y);``): the port declarations
    this parser reads from the body are then absent.
    """
    out: List[Tuple[str, str, bool]] = []
    pos = 0
    end_re = re.compile(r"\bendmodule\b")
    while True:
        h = _MODULE_HEADER_RE.search(text, pos)
        if not h:
            break
        name = h.group(1)
        body_start = h.end()
        e = end_re.search(text, body_start)
        if not e:
            break
        out.append((name, text[body_start:e.start()], bool(_ANSI_DIR_RE.search(h.group(0)))))
        pos = e.end()
    return out


def extract_ports(body: str) -> Tuple[Set[str], Dict[str, Tuple[int, int]], Dict[str, str]]:
    """Given a module body, return ``(port names, {vector port: (msb, lsb)},
    {port: direction})``.

    Bus widths and net-type qualifiers are stripped from the NAME set
    (e.g. ``output [3:0] state`` -> ``state``), but the declared range is kept
    separately: mapping a child's ``port[i]`` onto a parent-side connection
    EXPRESSION is positional, so the range is the only thing that says which
    element of the expression bit ``i`` is.

    A net-name appearing inside a port declaration AFTER a separator (comma)
    counts as an additional port.  Port lists may be re-declared multiple
    times in the body (DC commonly emits ``input [15:0] instr;`` followed by
    ``input clk, rst_n, ...;`` on different lines).
    """
    ports: Set[str] = set()
    ranges: Dict[str, Tuple[int, int]] = {}
    dirs: Dict[str, str] = {}
    for m in _PORT_DECL_RE.finditer(body):
        rhs = m.group(2)
        rng = _PORT_RANGE_RE.match(rhs.lstrip())
        this_range = ((int(rng.group(1)), int(rng.group(2))) if rng else None)
        rhs = re.sub(r"\[[^\]]+\]", " ", rhs)
        rhs = re.sub(r"\b(?:" + _NET_TYPE_KW + r"|signed|unsigned)\b", " ", rhs)
        for tok in rhs.split(","):
            tok = tok.strip()
            if not tok:
                continue
            # Take the last word (in case of leftover qualifiers slipped through)
            parts = tok.split()
            if not parts:
                continue
            cand = parts[-1]
            if re.match(r"^\w+$", cand):
                ports.add(cand)
                dirs[cand] = m.group(1)
                if this_range is not None:
                    ranges[cand] = this_range
    return ports, ranges, dirs


_SKIP_KW_RE = re.compile(
    r"^(input|output|inout|wire|reg|logic|assign|always|initial|"
    r"genvar|generate|endgenerate|if|for|case|endcase|parameter|"
    r"localparam|defparam|specify|endspecify|function|endfunction|"
    r"task|endtask)\b"
)
_INST_HEAD_RE = re.compile(r"^(\w+)\s+(\\?\S+)\s*\((.*)$")
_PIN_RE = re.compile(r"\.(\w+)\s*\(\s*([^)]*?)\s*\)")


def parse_instances(body: str) -> Tuple[List[Instance], List[str]]:
    """Line-walking instance parser.  Returns the instances in declaration order with
    their named-port connections (``.PIN(expr)``) as written, and the names of instances
    whose non-empty connection list has no named connection (positional binding, which
    is not modelled: their pin map would be empty)."""
    out: List[Instance] = []
    positional: List[str] = []

    def emit(celltype: str, instname: str, blob: str) -> None:
        pin2net = {pm.group(1): pm.group(2).strip() for pm in _PIN_RE.finditer(blob)}
        if not pin2net and blob.strip():
            positional.append(instname)
        out.append(Instance(instname, celltype, pin2net))

    lines, i = body.splitlines(), 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if _SKIP_KW_RE.match(line):
            i += 1
            continue
        m = _INST_HEAD_RE.match(line)
        if not m:
            i += 1
            continue
        celltype, instname, rest = m.group(1), m.group(2), m.group(3)
        if celltype == "module":
            i += 1
            continue
        buf: List[str] = []
        if rest and rest.strip():
            if ");" in rest:
                buf.append(rest.split(");")[0])
                emit(celltype, instname, "\n".join(buf))
                i += 1
                continue
            buf.append(rest)
        i += 1
        while i < len(lines) and ");" not in lines[i]:
            buf.append(lines[i])
            i += 1
        if i < len(lines):
            buf.append(lines[i].split(");")[0])
        emit(celltype, instname, "\n".join(buf))
        i += 1
    return out, positional


def parse_modules(text: str) -> Dict[str, ModuleDef]:
    """Parse every ``module ... endmodule`` block of a netlist (raw text).  Returns
    ``{module_name: ModuleDef}``."""
    text = strip_comments(text)
    out: Dict[str, ModuleDef] = {}
    for name, body, ansi in split_module_blocks(text):
        ports, ranges, dirs = extract_ports(body)
        instances, positional = parse_instances(body)
        nets = dict(ranges)
        for m in _NET_DECL_RE.finditer(body):
            msb, lsb = int(m.group(1)), int(m.group(2))
            for tok in m.group(3).split(","):
                tok = tok.strip()
                if re.match(r"^\w+$", tok):
                    nets[tok] = (msb, lsb)
        if name in out:
            raise BuildError(f"netlist defines module {name!r} twice")
        out[name] = ModuleDef(name=name, ports=ports, instances=instances,
                              port_ranges=ranges, net_ranges=nets, port_dirs=dirs,
                              n_assign=len(_ASSIGN_RE.findall(body)),
                              assigns=_ASSIGN_STMT_RE.findall(body), ansi_header=ansi,
                              positional=positional)
    return out


# ============================================================
# Port-connection EXPRESSIONS
# ============================================================
# A parent may connect a child's vector port with something other than a plain
# identifier -- DC emits a concatenation whenever it buffers a single bit of a
# bus on its way into a sub-module (`.src1({src1_reg[31:9], n1, src1_reg[7:0]})`).
# Verilog binds such an expression POSITIONALLY, so `src1[8]` is NOT
# "the connected net, bit 8"; it is the 8th bit counted from the expression's
# LSB end.  Gluing the child's `[8]` onto the parent's expression text invents a
# net that exists nowhere, and every net whose only path in is through that port
# silently leaves the graph.

def split_concat(expr: str) -> List[str]:
    """Top-level comma split of ``{a, b[3:0], c}`` -> ``['a', 'b[3:0]', 'c']``.

    Nested braces and bracketed ranges are respected, so a comma inside either
    never splits.  A non-concatenation is returned as a single element.
    """
    expr = expr.strip()
    if not (expr.startswith("{") and expr.endswith("}")):
        return [expr]
    inner = expr[1:-1]
    out: List[str] = []
    depth = 0
    cur: List[str] = []
    for ch in inner:
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        out.append(tail)
    return [t for t in out if t]


def expr_bits(expr: str,
              net_ranges: Optional[Dict[str, Tuple[int, int]]] = None
              ) -> Optional[List[str]]:
    """Flatten a port-connection expression into an MSB-FIRST list of bit refs.

    Returns ``None`` when the expression is a plain identifier -- the
    whole-vector case, where the child slices the parent's net itself and no
    positional mapping is needed.

    Refuses (returns ``None``) for anything it cannot flatten EXACTLY -- an
    unsized constant, a 2-D reference, a nested expression it does not model.
    Callers resolve that ambiguity with ``whole_conn_bits`` and raise rather than
    guess.

    ``net_ranges`` is the declared width of the ENCLOSING module's nets.  Without it
    a bare identifier inside a concatenation counts as ONE bit, because nothing here
    can know it is a bus: ``.ai({n134, dif_im_full})`` came out 2 bits wide against a
    16-bit port (``dif_im_full`` is ``[15:1]``, so the true width is 1 + 15).
    """
    expr = expr.strip()
    if not expr:
        return None
    if not (expr.startswith("{") or _SLICE_RE.match(expr)):
        return None                    # plain identifier / single bit / constant
    out: List[str] = []
    for elem in split_concat(expr):
        if elem.startswith("{"):
            sub = expr_bits(elem, net_ranges)
            if sub is None:
                return None
            out.extend(sub)
            continue
        sl = _SLICE_RE.match(elem)
        if sl:
            base, msb, lsb = sl.group(1), int(sl.group(2)), int(sl.group(3))
            step = -1 if msb >= lsb else 1
            out.extend(f"{base}[{i}]" for i in range(msb, lsb + step, step))
            continue
        if _SIZED_CONST_RE.match(elem):
            cb = _const_bits(elem)
            if cb is None:
                return None            # decimal -- not bit-addressable here
            out.extend(cb)
            continue
        if "[" in elem and not _BITREF_RE.match(elem):
            return None                # 2-D or otherwise unmodelled
        # A bare identifier: one bit ONLY if it is not a declared vector.
        if net_ranges is not None and elem in net_ranges:
            msb, lsb = net_ranges[elem]
            step = -1 if msb >= lsb else 1
            out.extend(f"{elem}[{i}]" for i in range(msb, lsb + step, step))
            continue
        out.append(elem)               # scalar net, or an explicit single bit
    return out


def _const_bits(elem: str) -> Optional[List[str]]:
    """A sized Verilog constant -> its own width, MSB-first.  ``None`` on decimal.

    Shared by the concatenation path and the whole-connection path so a constant is
    expanded the same way whether it appears INSIDE a concatenation or AS the whole
    connection of a vector port.
    """
    c = _SIZED_CONST_RE.match(elem)
    if not c:
        return None
    width, radix, digits = int(c.group(1)), c.group(2).lower(), c.group(3)
    per = {"b": 1, "o": 3, "h": 4}.get(radix)
    if per is None:                    # decimal -- not bit-addressable here
        return None
    bits = "".join(
        ("x" * per if ch in "xX?" else
         "z" * per if ch in "zZ" else
         bin(int(ch, 16 if radix == "h" else 8))[2:].zfill(per))
        for ch in digits.replace("_", ""))
    bits = bits[-width:].rjust(width, bits[0] if bits else "0")
    return [f"1'b{b}" for b in bits]


# A connection that is a single identifier: a plain net name, possibly a
# hierarchical path (`u_s0/dif_im_full`) or an escaped name.  Brackets, braces
# and commas are excluded because every one of those shapes is handled above.
_PLAIN_ID_RE = re.compile(r"^[^\[\]{},\s]+$")


def whole_conn_bits(conn: str,
                    conn_ranges: Optional[Dict[str, Tuple[int, int]]]
                    ) -> Optional[List[str]]:
    """MSB-first bit list for the connection shapes ``expr_bits`` declines.

    ``expr_bits`` returns ``None`` for THREE different situations:

      * a plain identifier            -> correct as "whole vector" ONLY if the parent's
                                         declared range numbers its bits the same way
                                         the child's port does;
      * a single bit reference `x[3]` -> gluing `[bit]` on top would produce the 2-D
                                         name `x[3][0]`, which names nothing;
      * anything unmodelled           -> a net named after the source text.

    Resolving them here makes the caller's positional arithmetic -- the SAME
    arithmetic the concatenation path uses -- apply to every shape, and turns the
    leftovers into a refusal instead of a guess.

    Returns ``None`` only when the shape cannot be resolved EXACTLY.
    """
    c = conn.strip()
    if _BITREF_RE.match(c):
        return [c]                     # one explicit bit, width 1
    if _SIZED_CONST_RE.match(c):
        return _const_bits(c)          # None on decimal -> caller refuses
    if c[:1].isdigit() or c[:1] == "'":
        return None                    # unsized/odd constant: width unknowable
    if _PLAIN_ID_RE.match(c):
        rng = (conn_ranges or {}).get(c)
        if rng is None:
            # No vector declaration anywhere for this name: in a gate-level
            # netlist that is a scalar, i.e. exactly one bit.  Saying so lets the
            # caller's width check fire when a wider port is bound to it.
            return [c]
        msb, lsb = rng
        step = -1 if msb >= lsb else 1
        return [f"{c}[{i}]" for i in range(msb, lsb + step, step)]
    return None                        # brace / unmodelled -> refuse


def register_net_ranges(mdef: ModuleDef,
                        inst_path: str,
                        registry: Dict[str, Tuple[int, int]]) -> None:
    """Record ``mdef``'s declared vector ranges under their GLOBAL net names.

    A child that bit-selects one of its ports needs the PARENT's declared range
    to place that bit, but by the time the child is resolved the connection has
    already been rewritten to a global name (`dif_im_full` -> `u_s0/dif_im_full`)
    and the parent's ModuleDef is out of scope.  Accumulating the ranges under
    the global names during the descent keeps that information reachable.

    Ports are skipped below the top: a port's global identity is whatever the
    parent connected to it, and the parent registered that net's range itself.
    At the top there is no parent, so the top's own ports resolve to themselves
    and must be registered here.
    """
    for net, rng in mdef.net_ranges.items():
        if inst_path and net in mdef.ports:
            continue
        registry[f"{inst_path}/{net}" if inst_path else net] = rng


def _port_bit_of_expr(port: str, bit: int, conn: str,
                      port_ranges: Dict[str, Tuple[int, int]],
                      inst_path: str,
                      conn_ranges: Optional[Dict[str, Tuple[int, int]]] = None
                      ) -> str:
    """``port[bit]`` resolved through a parent connection -- ANY connection.

    Verilog binds a port POSITIONALLY whatever the connection looks like, so a bare
    name whose declared range is `[16:1]` gives `ai[0]` the parent's bit **1**, not 0.

    The connection's bits are MSB-first (``expr_bits`` for an expression,
    ``whole_conn_bits`` for everything else); the child's declared range says
    where ``bit`` sits in that order.  Anything that does not line up EXACTLY --
    an unknown range, a width mismatch, an unflattenable element -- raises,
    because the alternative is a net name that names nothing and takes its whole
    downstream cone out of the injectable set without a single log line.
    """
    # conn_ranges are the PARENT's net widths -- exactly what a bare bus inside
    # a concatenation needs to be counted correctly.
    bits = expr_bits(conn, conn_ranges)
    rng = port_ranges.get(port)
    if bits is None:
        if rng is None:
            # A SCALAR port: there is no declared range, so there is no
            # positional question to answer.
            return f"{conn}[{bit}]"
        bits = whole_conn_bits(conn, conn_ranges)
        if bits is None:
            raise BuildError(
                f"[hier] {inst_path or '<top>'}: port '{port}' bit {bit} "
                f"is connected with {conn!r}, a shape this parser cannot flatten "
                f"exactly. Refusing to guess: gluing the bit index onto the "
                f"connection text would invent a net that names nothing and drop "
                f"its whole forward cone without a log line.")
    if rng is None:
        raise BuildError(
            f"[hier] {inst_path or '<top>'}: port '{port}' is connected "
            f"with the {len(bits)}-bit expression {conn!r}, but the sub-module "
            f"declares no range for it, so bit {bit} has no defined position.")
    msb, lsb = rng
    width = abs(msb - lsb) + 1
    if width != len(bits):
        raise BuildError(
            f"[hier] {inst_path or '<top>'}: port '{port}' is declared "
            f"[{msb}:{lsb}] ({width} bits) but the parent connects a "
            f"{len(bits)}-bit expression {conn!r}. Refusing to guess the alignment.")
    idx = (msb - bit) if msb >= lsb else (bit - msb)
    if not 0 <= idx < len(bits):
        raise BuildError(
            f"[hier] {inst_path or '<top>'}: bit {bit} is outside port "
            f"'{port}' range [{msb}:{lsb}]")
    return bits[idx]


def resolve_net(local_net: str,
                inst_path: str,
                module_ports: Set[str],
                parent_pin2net: Dict[str, str],
                port_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
                net_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
                conn_ranges: Optional[Dict[str, Tuple[int, int]]] = None) -> str:
    """Resolve a net referenced inside a sub-module instance to its global name.

      * If the net is one of the sub-module's ports, the resolved value is
        whatever the parent connected to that port.  An unconnected port keeps
        its local name.
      * Otherwise the net is a local wire; it is namespace-qualified with the
        instance path.

    Bus-bit references like ``n[3]`` compare the base identifier (``n``) to the
    port set; the qualified output keeps the bit suffix.

    A connection EXPRESSION (concatenation or slice) is resolved ELEMENT-WISE
    and handed back in canonical form, so the next level down sees global net
    names inside the braces and can index it positionally.
    """
    port_ranges = port_ranges or {}
    net_ranges = net_ranges or {}
    local_net = local_net.strip()

    # An expression: resolve every element in this scope and re-emit it, FULLY
    # EXPANDED to single bits.  Doing this BEFORE the bit-reference branch is
    # what keeps the information alive -- gluing the instance path onto the whole
    # `{...}` blob would make the elements unrecoverable downstream.  Expanding
    # here rather than lazily matters too: a bare VECTOR element
    # (`{1'b0, top_state}`) has a width only its declaring module knows, and this
    # is the last scope that knows it.
    if local_net.startswith("{"):
        bits: List[str] = []
        for e in split_concat(local_net):
            rng = net_ranges.get(e)
            if rng is not None and not e.startswith("{"):
                msb, lsb = rng
                step = -1 if msb >= lsb else 1
                bits.extend(
                    resolve_net(f"{e}[{i}]", inst_path, module_ports,
                                parent_pin2net, port_ranges, net_ranges,
                                conn_ranges)
                    for i in range(msb, lsb + step, step))
            else:
                r = resolve_net(e, inst_path, module_ports, parent_pin2net,
                                port_ranges, net_ranges, conn_ranges)
                # An already-canonical sub-expression is spliced in flat, so the
                # result is always a one-level list of bits.
                bits.extend(split_concat(r) if r.startswith("{") else [r])
        return "{" + ", ".join(bits) + "}"

    sl = _SLICE_RE.match(local_net)
    if sl:
        base, msb, lsb = sl.group(1), int(sl.group(2)), int(sl.group(3))
        if base in module_ports and base in parent_pin2net:
            step = -1 if msb >= lsb else 1
            elems = [_port_bit_of_expr(base, i, parent_pin2net[base],
                                       port_ranges, inst_path, conn_ranges)
                     for i in range(msb, lsb + step, step)]
            # A slice of a whole-vector connection keeps its compact form ONLY
            # when the parent numbers those bits exactly as the child does.
            # Emitting `parent[msb:lsb]` unconditionally would put the CHILD's
            # indices on the PARENT's name.  The test is the resolved bits
            # themselves, so it cannot drift from what the bit path decided.
            conn = parent_pin2net[base]
            if (expr_bits(conn) is None
                    and elems == [f"{conn}[{i}]"
                                  for i in range(msb, lsb + step, step)]):
                return f"{conn}[{msb}:{lsb}]"
            return "{" + ", ".join(elems) + "}"
        if base in module_ports:
            return local_net              # unconnected -- keep as-is
        return f"{inst_path}/{local_net}" if inst_path else local_net

    bus_m = _BITREF_RE.match(local_net)
    if bus_m:
        base = bus_m.group(1)
        bit = int(bus_m.group(2))
        if base in module_ports:
            if base in parent_pin2net:
                return _port_bit_of_expr(base, bit, parent_pin2net[base],
                                         port_ranges, inst_path, conn_ranges)
            return f"{base}[{bit}]"       # unconnected -- keep as-is
        if inst_path:
            return f"{inst_path}/{base}[{bit}]"
        return f"{base}[{bit}]"

    # Plain identifier
    if local_net in module_ports:
        return parent_pin2net.get(local_net, local_net)
    # Constants (1'b0, 1'b1, etc) and tied nets stay verbatim
    if re.match(r"^\d", local_net) or local_net in ("", "1'b0", "1'b1"):
        return local_net
    if inst_path:
        return f"{inst_path}/{local_net}"
    return local_net


# ============================================================
# Hierarchical elaboration
# ============================================================
def _assign_side_bits(expr: str, mdef: ModuleDef) -> Optional[List[str]]:
    """One side of an ``assign`` as an MSB-first list of local bits (constants as
    ``1'b0``/``1'b1``/``1'bx``/``1'bz``).  ``None`` when it is not a plain net
    expression (e.g. it contains logic operators)."""
    e = expr.strip()
    if _SIZED_CONST_RE.match(e):
        return _const_bits(e)
    # every element must be a net name, a bit or slice of one, or a sized constant
    for elem in (split_concat(e) if e.startswith("{") else [e]):
        if elem.startswith("{"):
            if _assign_side_bits(elem, mdef) is None:
                return None
            continue
        base = re.sub(r"\[[^\]]*\]$", "", elem)
        if not (_VERILOG_ID_RE.match(base) or _SIZED_CONST_RE.match(elem)):
            return None
    bits = expr_bits(e, mdef.net_ranges)
    if bits is not None:
        return bits
    if _BITREF_RE.match(e):
        return [e]
    if _VERILOG_ID_RE.match(e):
        rng = mdef.net_ranges.get(e)
        if rng is None:
            return [e]
        msb, lsb = rng
        step = -1 if msb >= lsb else 1
        return [f"{e}[{i}]" for i in range(msb, lsb + step, step)]
    return None


_CONST_BIT_RE = re.compile(r"^1'b[01xXzZ]$")
_VERILOG_ID_RE = re.compile(r"^(?:[A-Za-z_][\w$]*|\\\S+)$")


@dataclass
class Elaboration:
    """Result of walking the hierarchy.

    leaves      -- {path: Instance} for every library-cell instance AND every
                   non-library instance (sub-module instances and black boxes are
                   recorded too, before their children, with their parent-side pin
                   nets: the graph builder needs the nets that touch them)
    blackboxes  -- paths of opaque-macro instances (``library.blackbox_cells``)
    n_assign    -- `assign` statements over all elaborated module instances
    n_alias_bits -- assigned bits merged into their right-hand net
    n_tied_bits  -- assigned bits driven by a constant (left undriven in the graph)
    """
    leaves: Dict[str, Instance]
    blackboxes: List[str]
    n_assign: int
    n_alias_bits: int = 0
    n_tied_bits: int = 0
    aliases: Dict[str, str] = field(default_factory=dict)   # merged net -> surviving net


def elaborate(modules: Dict[str, ModuleDef], top: str, lib_celltypes: Set[str],
              is_blackbox: Callable[[str], bool]) -> Elaboration:
    """Recursively expand the module hierarchy starting at ``top``.

    Returns every instance keyed by hierarchical path, each ``Instance.pin2net`` being
    the resolved GLOBAL net mapping (sub-module ports rewritten to the parent-side
    net; local wires namespace-qualified by instance path).

    An instance whose cell type is neither a library cell, nor a module of this
    netlist, nor a declared black box raises: it would otherwise be an opaque leaf
    whose outputs drive nothing the graph knows about.
    """
    if top not in modules:
        raise BuildError(f"design.top = {top!r}: no module of that name in design.netlist "
                         f"(its modules: {', '.join(sorted(modules)[:10])}"
                         f"{', ...' if len(modules) > 10 else ''})")

    leaves: Dict[str, Instance] = {}
    blackboxes: List[str] = []
    counts = {"assign": 0, "alias_bits": 0, "tied_bits": 0}
    alias_of: Dict[str, str] = {}          # left-hand global net -> right-hand global net
    # {global net name: declared (msb, lsb)} accumulated as the descent goes.
    # Filled before a module's instances are resolved, so a child always finds
    # the range of whatever its parent bound to its ports.
    net_ranges_global: Dict[str, Tuple[int, int]] = {}

    def descend(module_name: str, inst_path: str, parent_pin2net: Dict[str, str]) -> None:
        mdef = modules[module_name]
        where = f"module {module_name!r} (at {inst_path or '<top>'})"
        if mdef.ansi_header:
            raise BuildError(f"{where} declares its port directions in the header "
                             f"(ANSI style); only body declarations are supported")
        if mdef.positional:
            raise BuildError(f"{where}: instance(s) {mdef.positional[:5]} are connected by "
                             f"position; only named connections (.PIN(net)) are supported")
        counts["assign"] += mdef.n_assign
        register_net_ranges(mdef, inst_path, net_ranges_global)
        for lhs, rhs in mdef.assigns:
            lb, rb = _assign_side_bits(lhs, mdef), _assign_side_bits(rhs, mdef)
            if lb is None or rb is None:
                raise BuildError(
                    f"{where}: `assign {lhs} = {rhs};` is not a net alias (only nets, "
                    f"slices, concatenations and constants are supported)")
            # Verilog pairs the two sides from the LSB; missing RHS bits are 0.
            rb = ["1'b0"] * max(0, len(lb) - len(rb)) + rb[max(0, len(rb) - len(lb)):]
            for lbit, rbit in zip(lb, rb):
                if _CONST_BIT_RE.match(lbit):
                    raise BuildError(f"{where}: `assign {lhs} = ...` assigns to a constant")
                L = resolve_net(lbit, inst_path, mdef.ports, parent_pin2net,
                                mdef.port_ranges, mdef.net_ranges, net_ranges_global)
                R = rbit if _CONST_BIT_RE.match(rbit) else resolve_net(
                    rbit, inst_path, mdef.ports, parent_pin2net, mdef.port_ranges,
                    mdef.net_ranges, net_ranges_global)
                if _CONST_BIT_RE.match(R) or L.startswith("{") or R.startswith("{"):
                    counts["tied_bits"] += 1
                    continue
                if L == R:
                    continue
                if L in alias_of and alias_of[L] != R:
                    raise BuildError(f"{where}: net {L!r} is assigned twice")
                alias_of[L] = R
                counts["alias_bits"] += 1
        for inst in mdef.instances:
            child_path = f"{inst_path}/{inst.name}" if inst_path else inst.name
            resolved = {
                pin: resolve_net(net, inst_path, mdef.ports, parent_pin2net,
                                 mdef.port_ranges, mdef.net_ranges,
                                 net_ranges_global)
                for pin, net in inst.pin2net.items()
            }
            if child_path in leaves:
                raise BuildError(f"instance path {child_path!r} is declared twice")
            # Non-library instances (sub-modules, black boxes) are recorded as well:
            # the graph builder collects the nets touching them (non-library outputs
            # seed the forward region).
            leaves[child_path] = Instance(name=child_path, celltype=inst.celltype,
                                          pin2net=resolved)
            if inst.celltype in lib_celltypes:
                continue
            if inst.celltype in modules:
                descend(inst.celltype, child_path, resolved)
            elif is_blackbox(inst.celltype):
                blackboxes.append(child_path)
            else:
                raise BuildError(
                    f"instance {child_path!r} has cell type {inst.celltype!r}, which is "
                    f"not defined in library.cell_functions, not a module of this netlist, "
                    f"and not declared as a black box (library.blackbox_cells), so its "
                    f"function is unknown.")

    # The top module has no parent context; its ports are its primary IO and map
    # to themselves.
    top_def = modules[top]
    descend(top, "", {p: p for p in top_def.ports})

    if alias_of:
        def find(x: str) -> str:
            seen = []
            while x in alias_of:
                seen.append(x)
                x = alias_of[x]
                if len(seen) > len(alias_of):
                    raise BuildError(f"`assign` statements form a loop through {seen[0]!r}")
            for y in seen:                     # path compression
                alias_of[y] = x
            return x

        def rename(v: str) -> str:
            if v.startswith("{"):
                return "{" + ", ".join(find(b) for b in split_concat(v)) + "}"
            return find(v)

        for inst in leaves.values():
            inst.pin2net = {p: rename(n) for p, n in inst.pin2net.items()}
    return Elaboration(leaves=leaves, blackboxes=blackboxes, n_assign=counts["assign"],
                       n_alias_bits=counts["alias_bits"], n_tied_bits=counts["tied_bits"],
                       aliases={k: find(k) for k in list(alias_of)} if alias_of else {})


def primary_input_bits(top_def: ModuleDef) -> Set[str]:
    """The top module's input ports, expanded to bit names."""
    out: Set[str] = set()
    for p, d in top_def.port_dirs.items():
        if d != "input":
            continue
        rng = top_def.port_ranges.get(p)
        if rng is None:
            out.add(p)
        else:
            msb, lsb = rng
            step = -1 if msb >= lsb else 1
            out.update(f"{p}[{i}]" for i in range(msb, lsb + step, step))
    return out


def crosscheck_sdf(leaves: Dict[str, Instance], blackboxes: List[str],
                   sdf_inst_celltype: Dict[str, str], lib_celltypes: Set[str],
                   is_untimed: Callable[[str], bool]) -> Dict[str, object]:
    """Netlist leaves vs SDF ``(INSTANCE ...)`` blocks.

    * ``missing``: the SDF annotates an instance the netlist does not have;
    * ``extra``:   a library-cell leaf that has timing arcs (not ``is_untimed``) but no
                   SDF block -- its arcs would silently fail to join;
    * ``celltype_mismatch``: same path, different cell type in the two files.

    Untimed cells (no timing arc: tie cells, bus holders, ...) are never required in
    the SDF, but an SDF block for one is not an error either.
    """
    sdf_paths = {p for p in sdf_inst_celltype if p != "*"}
    known = {p for p, inst in leaves.items() if inst.celltype in lib_celltypes}
    known |= set(blackboxes)
    timed = {p for p in known
             if p in leaves and leaves[p].celltype in lib_celltypes
             and not is_untimed(leaves[p].celltype)}
    missing = sorted(sdf_paths - known)
    extra = sorted(timed - sdf_paths)
    mismatch = sorted(p for p in (sdf_paths & known)
                      if sdf_inst_celltype[p] != leaves[p].celltype)
    return {
        "n_sdf_instances": len(sdf_paths),
        "n_timed_leaves": len(timed),
        "n_untimed_leaves": len(known) - len(timed) - len(blackboxes),
        "n_blackboxes": len(blackboxes),
        "missing": missing,
        "extra": extra,
        "celltype_mismatch": mismatch,
        "ok": not (missing or extra or mismatch),
    }
