"""Adapt a testbench written for the RTL so it simulates the gate-level netlist.

Most testbenches need no source edits: only the file list changes, from
``<rtl...> + <tb>`` to ``<netlist> + <cell models> + <tb>``.  Edits are needed
when the testbench

* peeks at internal signals (``dut.<inst>.<reg>``) that synthesis flattened or
  renamed -- fixed by ``hier_substitutions`` (``{regex, replacement}`` applied
  in order to the testbench text) and, optionally, rules auto-derived from the
  netlist (``auto_derive_hier_substitutions``; manual rules stay authoritative);
* relies on memory macros that synthesis left as empty black-box stubs
  (``macro_rtl_subs``: behavioural RTL per macro; parameter-uniquified stub
  clones are aliased to it by a generated wrapper and stripped from the netlist
  copy used for simulation), or on modules that must be replaced by RTL
  (``netlist_strip_modules`` + ``netlist_substitute_files``).

The entry point is :func:`adapt_testbench`; it writes the adapted files and a
``result.json`` report into its output directory.
"""
from __future__ import annotations

import datetime as _dt
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from .errors import WorkloadError

TOOL_NAME = "setfi record (testbench adaptation)"
from .. import __version__ as TOOL_VERSION  # noqa: E402


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _resolve_files(paths: Sequence, kind: str, error_kind: str = "config_invalid") -> List[Path]:
    out: List[Path] = []
    missing: List[str] = []
    for p in paths:
        rp = Path(p).resolve()
        if not rp.is_file():
            missing.append(str(rp))
        out.append(rp)
    if missing:
        raise WorkloadError(error_kind, f"{kind}: file(s) not found: {missing}")
    return out


def _apply_substitutions(tb_text: str, subs: List[dict]):
    """Apply each substitution in order. Returns (new_text, applied_log)
    where applied_log records {regex, replacement, n_subs}."""
    applied: List[dict] = []
    text = tb_text
    for s in subs:
        if not isinstance(s, dict) or "regex" not in s or "replacement" not in s:
            raise WorkloadError("config_invalid",
                                "each workload.tb_substitutions entry needs "
                                "{regex, replacement}; got " + repr(s))
        try:
            pat = re.compile(s["regex"])
        except re.error as e:
            raise WorkloadError("config_invalid",
                                f"workload.tb_substitutions: bad regex {s['regex']!r}: {e}") from e
        text, n = pat.subn(s["replacement"], text)
        applied.append({"regex": s["regex"], "replacement": s["replacement"], "n_subs": n})
    return text, applied


# ---------------------------------------------------------------------------
# Opt-in auto-derivation of hier substitutions.
#
# When DC's `compile_ultra` flattens the design (`syn_flatten=true`), TB
# cross-module references like ``dut.<inner>.<sig>[idx]`` no longer resolve
# — DC has rewritten them into top-scope flat wires (commonly
# ``<inner>_<sig>``). Authoring the substitution rules by hand is
# labor-intensive and error-prone (e.g. the MSB-first `(N-1-(\1))` flip).
#
# This helper scans the TB for ``dut.<inner>.<sig>`` (and indexed
# ``dut.<inner>.<sig>[idx]``) references, then for each such reference
# searches the mapped.v top module for a flat wire/reg declaration named
# ``<inner>_<sig>``. Scalar refs (no `[idx]`) get a clean rename;
# **indexed** refs get a candidate slice rule where element width is
# guessed from the rtl_dut_files (looking up the original
# ``<inner>.<sig>`` declaration). If multiple candidates match or the
# element width can't be inferred unambiguously, the auto-deriver emits a
# `hier_substitution_ambiguous` error rather than silently picking one
# (refuse rather than guess).
#
# Default behavior: opt-in via ``auto_derive_hier_substitutions``
# (boolean, default false). Auto-derived rules are APPENDED to any explicit
# ``hier_substitutions`` so manual rules remain authoritative.
# ---------------------------------------------------------------------------


# XMR shape: ``<dut>.<inner_1>.<inner_2>...<inner_N>.<sig>[<idx>?]`` with at
# least two dotted tokens after ``<dut>`` (i.e. one or more intermediate
# instance segments + the leaf signal). The ``{2,}`` minimum preserves the
# 0.1.x assumption that 2-token refs like ``dut.<sig>`` are not in scope for
# auto-derive (the resolver only fires for hierarchical peeks). Anything
# deeper — ``dut.u_dut.u_rf.mem`` for a netlist wrapped in an extra layer, etc. —
# is matched greedily and split by the resolver into (inner_chain, sig).
_TB_XMR_RE = re.compile(
    r"\b(?P<dut>[A-Za-z_][A-Za-z0-9_$]*)"
    r"(?P<path>(?:\.[A-Za-z_][A-Za-z0-9_$]*){2,})"
    r"(?P<idx>\[[^\]]+\])?",
)


def _grep_top_module_flat_wires(mapped_text: str, top_module: str
                                  ) -> dict[str, str]:
    """Pull a {<flat_name>: '<msb>:<lsb>'|'<scalar>'} map from the top module
    of `mapped_text`. We bound the search to the top module body so we
    don't pick up sub-module-internal names. Returns empty dict if the top
    isn't located (we then fall back to no auto-derive rather than guess).
    """
    pat = re.compile(
        rf"\bmodule\s+{re.escape(top_module)}\b[^;]*;\s*(?P<body>.*?)\bendmodule\b",
        re.DOTALL,
    )
    m = pat.search(mapped_text)
    if not m:
        return {}
    body = m.group("body")
    flat: dict[str, str] = {}
    decl_re = re.compile(
        r"^\s*(?:wire|reg|logic|bit|tri)\s+"
        r"(?P<range>\[\s*[^\]]+?\s*\])?"
        r"\s*(?P<names>[A-Za-z_][A-Za-z0-9_$,\s]*?)\s*;",
        re.MULTILINE,
    )
    for d in decl_re.finditer(body):
        rng = d.group("range") or ""
        for nm in (n.strip() for n in d.group("names").split(",")):
            if nm and nm not in flat:
                flat[nm] = rng or "scalar"
    return flat


def _grep_inner_signal_widths(rtl_dut_files: list[Path]
                                ) -> dict[tuple[str, str], int]:
    """Best-effort: for ``logic [W-1:0] <sig> [0:N-1];`` declarations
    inside any rtl_dut_files, return {(<module_name>, <sig>): <element bit
    width>}. Used to compute the per-index slice for indexed XMRs. We don't
    parse <expressions>; only literal-bound widths get recorded.
    """
    widths: dict[tuple[str, str], int] = {}
    mod_re = re.compile(
        r"\bmodule\s+(?P<m>[A-Za-z_][A-Za-z0-9_$]*)\b.*?\bendmodule\b",
        re.DOTALL,
    )
    arr_re = re.compile(
        r"\b(?:logic|wire|reg|bit)\s*\[\s*(?P<top>[^\]]+?)\s*:\s*0\s*\]"
        r"\s+(?P<sig>[A-Za-z_][A-Za-z0-9_$]*)\s*"
        r"\[\s*0\s*:\s*[^\]]+?\s*\]\s*;",
    )
    for f in rtl_dut_files:
        try:
            text = f.read_text()
        except OSError:
            continue
        for mm in mod_re.finditer(text):
            mod = mm.group("m")
            body = mm.group(0)
            for am in arr_re.finditer(body):
                # Resolve `<top>` to literal width if it's of the form
                # ``<NAME>-1`` or just a digit.
                t = am.group("top").strip()
                lit = re.match(r"-?\d+\s*$", t)
                if lit:
                    widths[(mod, am.group("sig"))] = int(lit.group(0)) + 1
                else:
                    sub = re.match(
                        r"([A-Za-z_][A-Za-z0-9_$]*)\s*-\s*1\s*$", t)
                    if sub:
                        # Look for `parameter <NAME> = <int>` in the same
                        # module body to resolve.
                        pname = sub.group(1)
                        pm = re.search(
                            rf"\bparameter\s+(?:int\s+)?{re.escape(pname)}\s*=\s*"
                            r"(?P<v>-?\d+)\b",
                            body,
                        )
                        if pm:
                            widths[(mod, am.group("sig"))] = int(pm.group("v"))
    return widths


def _grep_inner_module_for_inst(
    rtl_dut_files: list[Path], inner_inst: str,
) -> str | None:
    """Find the module that an instance ``<inner_inst>`` belongs to by grep'ing
    rtl_dut_files for ``<MOD>\\s+(#( ... ))?\\s+<inner_inst>\\s*\\(``.
    Returns the module name or None if ambiguous / not found.

    Uses a manual paren-balanced skip for the optional ``#( ... )`` parameter
    block so that ``regbank #(.ADDR_W(REG_AW), .DATA_W(DATA_W)) u_rb (`` —
    which contains nested parens — is matched (the v0.1 flat regex with
    ``[^)]*`` couldn't span those).
    """
    KEYWORDS = {"if", "else", "for", "case", "begin", "end",
                 "module", "endmodule", "always", "always_ff", "always_comb",
                 "assign", "wire", "logic", "reg", "input", "output",
                 "inout", "generate", "endgenerate", "function",
                 "endfunction", "task", "endtask", "return", "while",
                 "do", "default", "endcase", "parameter", "localparam",
                 "typedef", "struct", "union", "enum", "package",
                 "endpackage", "import", "export", "interface",
                 "endinterface", "modport", "class", "endclass", "new",
                 "extends", "implements", "virtual", "pure", "ref",
                 "const", "static", "automatic", "rand", "randc",
                 "constraint", "covergroup", "endgroup", "property",
                 "endproperty", "sequence", "endsequence",
                 "initial", "final", "fork", "join", "join_any",
                 "join_none", "wait", "disable", "force", "release",
                 "repeat", "forever"}
    name_id = r"[A-Za-z_][A-Za-z0-9_$]*"
    head_re = re.compile(rf"\b(?P<mod>{name_id})\s*(?P<rest>#|{re.escape(inner_inst)}\s*\()")
    candidates: set[str] = set()
    for f in rtl_dut_files:
        try:
            text = f.read_text()
        except OSError:
            continue
        n = len(text)
        for m in head_re.finditer(text):
            cand = m.group("mod")
            if cand in KEYWORDS or cand == inner_inst:
                continue
            j = m.end("mod")
            while j < n and text[j].isspace():
                j += 1
            # Optional parameter block.
            if j < n and text[j] == "#":
                j += 1
                while j < n and text[j].isspace():
                    j += 1
                if j >= n or text[j] != "(":
                    continue
                # Paren-balance.
                depth = 1
                j += 1
                while j < n and depth > 0:
                    ch = text[j]
                    if ch == "(":
                        depth += 1
                    elif ch == ")":
                        depth -= 1
                    j += 1
                if depth != 0:
                    continue
                while j < n and text[j].isspace():
                    j += 1
            # Now match the instance identifier + opening paren.
            ident_m = re.match(rf"{re.escape(inner_inst)}\s*\(", text[j:])
            if not ident_m:
                continue
            candidates.add(cand)
    if len(candidates) == 1:
        return candidates.pop()
    return None


def _grep_module_body_signals(mapped_text: str, module_name: str
                              ) -> dict[str, str]:
    """Pull a {<sig>: '<msb>:<lsb>'|'scalar'} map from the named module's body
    in ``mapped_text``. Used by the keep-hier auto-derive second pass so XMRs into a sub-module body like
    ``dut.u_fifo.mem`` / ``dut.u_fifo.wptr`` can be resolved when DC ran in
    keep-hierarchy mode and the signal lives INSIDE the submod body rather
    than as a flat top-scope wire. Returns empty dict if the module body
    isn't located.
    """
    pat = re.compile(
        rf"\bmodule\s+{re.escape(module_name)}\b[^;]*;\s*(?P<body>.*?)\bendmodule\b",
        re.DOTALL,
    )
    m = pat.search(mapped_text)
    if not m:
        return {}
    body = m.group("body")
    sigs: dict[str, str] = {}
    decl_re = re.compile(
        r"^\s*(?:wire|reg|logic|bit|tri)\s+"
        r"(?P<range>\[\s*[^\]]+?\s*\])?"
        r"\s*(?P<names>[A-Za-z_][A-Za-z0-9_$,\s]*?)\s*;",
        re.MULTILINE,
    )
    for d in decl_re.finditer(body):
        rng = d.group("range") or ""
        for nm in (n.strip() for n in d.group("names").split(",")):
            if nm and nm not in sigs:
                sigs[nm] = rng or "scalar"
    return sigs


def _grep_module_submod_instances(mapped_text: str, module_name: str
                                    ) -> dict[str, str]:
    """Return ``{<inner_inst_name>: <submodule_module_name>}`` for every
    sub-module instance found inside the named module body of ``mapped_text``.

    Generic over the named module — the top-module variant just delegates
    here. Used by the multi-level XMR chain walker so that an XMR like
    ``dut.u_dut.u_rf.mem`` can be resolved by walking ``u_dut`` (in the
    wrapper top), then ``u_rf`` (in the submod ``u_dut`` instantiates), and
    finally looking up ``mem`` as a signal in the innermost submod body.
    """
    pat = re.compile(
        rf"\bmodule\s+{re.escape(module_name)}\b[^;]*;\s*(?P<body>.*?)\bendmodule\b",
        re.DOTALL,
    )
    m = pat.search(mapped_text)
    if not m:
        return {}
    body = m.group("body")
    out: dict[str, str] = {}
    # Match  `<TYPE_ID> <INST_ID> (`  at a line start (optional whitespace).
    # Skip lines that begin with reserved keywords (assign, wire, reg, ...).
    KEYWORDS = {
        "assign", "wire", "reg", "logic", "bit", "tri", "always",
        "always_ff", "always_comb", "always_latch", "if", "else",
        "for", "while", "do", "case", "casex", "casez", "endcase",
        "begin", "end", "module", "endmodule", "generate", "endgenerate",
        "function", "endfunction", "task", "endtask", "initial", "final",
        "input", "output", "inout", "parameter", "localparam",
        "typedef", "import", "export", "include", "specify", "endspecify",
        "primitive", "endprimitive", "fork", "join", "join_any", "join_none",
        "default", "return", "break", "continue", "force", "release",
    }
    inst_re = re.compile(
        r"^\s*(?P<type>[A-Za-z_][A-Za-z0-9_$]*)\s+"
        r"(?P<inst>[A-Za-z_][A-Za-z0-9_$]*)\s*\(",
        re.MULTILINE,
    )
    for im in inst_re.finditer(body):
        t = im.group("type")
        i = im.group("inst")
        if t in KEYWORDS:
            continue
        # First-seen wins; duplicate keys would mean the same instance name
        # appearing in multiple instantiation patterns, which RTL semantics
        # forbid anyway.
        out.setdefault(i, t)
    return out


def _grep_top_module_submod_instances(mapped_text: str, top_module: str
                                      ) -> dict[str, str]:
    """Back-compat alias for the generic submod-instance grep, scoped to the
    top module body. Older call sites only need the top scope (the
    auto-derive path used 1-deep keep-hier resolution); the chain walker
    introduced for layered wrappers calls ``_grep_module_submod_instances``
    directly with the current submod name per chain step.
    """
    return _grep_module_submod_instances(mapped_text, top_module)


def _walk_inst_chain(
    mapped_text: str, start_module: str, inst_chain: tuple[str, ...],
) -> tuple[str, list[tuple[str, str, str]]] | tuple[None, str]:
    """Walk an instance chain through nested module bodies of ``mapped_text``.

    Given ``start_module`` (e.g. ``top_wrapper``) and
    ``inst_chain = ("u_dut", "u_rf")``, step into the module body for each
    instance in turn and return the final submodule name (e.g. ``regfile``)
    plus a trace ``[(parent_module, inst_name, child_module), ...]`` for the
    rule's _detail.

    Returns ``(None, reason_str)`` on failure (instance not found inside the
    current submod body, or submod body itself missing from mapped.v).

    Used by the resolver for XMRs that traverse a wrapper instance layer the
    netlist preserves but the original DUT RTL doesn't have (layered
    wrappers introduce ``u_dut`` between the TB's ``dut`` and the original DUT
    submod chain; chains can be N levels deep in principle).
    """
    current = start_module
    trace: list[tuple[str, str, str]] = []
    for inst in inst_chain:
        children = _grep_module_submod_instances(mapped_text, current)
        if not children:
            return None, (f"submodule `{current}` body absent from mapped.v "
                          f"(or has no sub-instances); cannot continue chain "
                          f"walk into `{inst}`")
        child = children.get(inst)
        if child is None:
            return None, (f"instance `{inst}` not found in `{current}` body "
                          f"(saw {len(children)} sub-instance(s): "
                          f"{sorted(children.keys())[:8]}{'...' if len(children) > 8 else ''})")
        trace.append((current, inst, child))
        current = child
    return current, trace


def _auto_derive_hier_substitutions(
    tb_text: str, mapped_text: str, top_module: str,
    rtl_dut_files: list[Path], dut_inst_name: str,
) -> tuple[list[dict], list[dict]]:
    """Returns (rules, ambiguous). Rules are
    ``[{regex, replacement, _autogen: true, _kind, _detail}, ...]`` ready to
    feed into the existing substitution pass. ``ambiguous`` lists XMRs we
    refused to bind (multiple candidates / unknown widths) — the caller
    raises `hier_substitution_ambiguous` if non-empty.

    Two-pass behavior:

      Pass 1 (DC-flatten / ungroup convention) — look for a top-scope flat
      wire named ``<inner_1>_<inner_2>_..._<sig>`` (chain joined by ``_``).
      If present, emit either a scalar-rename rule or an MSB-first slicing
      rule (MSB-first packing).

      Pass 2 (keep-hierarchy convention) — if Pass 1's flat lookup fails,
      walk the instance chain ``(inner_1, inner_2, ..., inner_N)`` through
      nested module bodies in mapped.v (each step resolves an instance to
      its submodule type), then scan the innermost submod's body for
      ``<sig>``:
        - If ``<sig>`` exists in the submod body and the XMR is not indexed,
          emit a **no-op rule** (the keep-hier XMR resolves natively in VCS).
        - If ``<sig>`` exists as a packed flat vector (e.g.
          ``wire [127:0] mem;``) and the XMR IS indexed, derive element
          width from RTL and emit an MSB-first slicing rule with the
          left-hand path kept hierarchical (``dut.<chain>.<sig>[i] ->
          dut.<chain>.<sig>[(N-1-i)*W +: W]``).

    The chain walk handles wrapper-layer hierarchies introduced by layered
    wrappers (e.g. a redundancy wrapper around the design exposes an extra
    ``u_dut`` instance between the TB's ``dut`` and the original DUT submod
    tree; chains of arbitrary depth resolve the same way). For a 1-level
    chain (``dut.<inner>.<sig>``) behavior is byte-identical to the v0.1
    flat / keep-hier paths.

    Pass 2 is essential for keep-hierarchy netlists (``dut.u_fifo.mem[i]``,
    ``dut.u_fifo.wptr``) AND for layered wrappers (``dut.u_dut.u_rf.mem[i]``).
    """
    flat_wires = _grep_top_module_flat_wires(mapped_text, top_module)
    inner_widths = _grep_inner_signal_widths(rtl_dut_files)
    # Keep-hier second-pass support data.
    top_inst_to_submod = _grep_top_module_submod_instances(mapped_text, top_module)

    # Discover unique XMRs in the TB for this DUT instance only. Key by the
    # full instance chain tuple so deep wrapper paths
    # (``dut.u_dut.u_rf.mem``) collapse correctly with 3-level peers that
    # share an inner segment.
    seen: dict[tuple[tuple[str, ...], str, bool], None] = {}
    for m in _TB_XMR_RE.finditer(tb_text):
        if m.group("dut") != dut_inst_name:
            continue
        # path is ``.tok1.tok2....tokN.sig``; strip leading ``.`` and split.
        path_tokens = m.group("path").lstrip(".").split(".")
        if len(path_tokens) < 2:
            # Guarded by the regex {2,} quantifier, but defensive — a 1-token
            # match would be ``dut.sig`` which the resolver is not in scope for.
            continue
        inner_chain = tuple(path_tokens[:-1])
        sig = path_tokens[-1]
        seen.setdefault((inner_chain, sig, bool(m.group("idx"))), None)

    # If neither flat_wires (flattened) nor top_inst_to_submod (kept hierarchy)
    # gave anything, the top module was not located: nothing can be derived.
    if not flat_wires and not top_inst_to_submod:
        return [], []

    # Cache submod-body signal scans so we don't re-grep mapped.v for each
    # XMR pointing at the same submod.
    _submod_sig_cache: dict[str, dict[str, str]] = {}

    def _submod_sigs(name: str) -> dict[str, str]:
        s = _submod_sig_cache.get(name)
        if s is None:
            s = _grep_module_body_signals(mapped_text, name)
            _submod_sig_cache[name] = s
        return s

    rules: list[dict] = []
    ambiguous: list[dict] = []
    for (inner_chain, sig, is_indexed), _ in seen.items():
        chain_dotted = ".".join(inner_chain)      # "u_rf"   /   "u_dut.u_rf"
        chain_flat = "_".join(inner_chain)         # "u_rf"   /   "u_dut_u_rf"
        xmr_display = (f"{dut_inst_name}.{chain_dotted}.{sig}"
                       f"{'[idx]' if is_indexed else ''}")
        flat_name = f"{chain_flat}_{sig}"
        # Regex/replacement builders that work for any chain length. For the
        # 1-inner case these produce strings byte-identical to the v0.1
        # `dut\.<inner>\.<sig>` shape.
        path_escaped = re.escape(f"{dut_inst_name}.{chain_dotted}.{sig}")
        regex_nob = path_escaped + r"\b"
        regex_idx = path_escaped + r"\[([^\]]+)\]"
        kept_path = f"{dut_inst_name}.{chain_dotted}.{sig}"
        flat_path = f"{dut_inst_name}.{flat_name}"

        if flat_name not in flat_wires:
            # Pass 2: keep-hier — walk the instance chain through mapped.v.
            # The 1-element chain case is byte-equivalent to the v0.1
            # ``top_inst_to_submod.get(inner)`` path because
            # ``_walk_inst_chain(_, top_module, (inner,))`` just runs that
            # same lookup once.
            walk_result, walk_info = _walk_inst_chain(
                mapped_text, top_module, inner_chain,
            )
            if walk_result is None:
                # Neither convention recognizes this XMR.
                ambiguous.append({
                    "xmr": xmr_display,
                    "reason": (
                        f"no flat wire `{flat_name}` in top-module body "
                        f"(searched {len(flat_wires)} top-scope nets) AND "
                        f"keep-hier chain walk failed: {walk_info}"
                    ),
                })
                continue
            submod = walk_result
            submod_sigs = _submod_sigs(submod)
            if sig not in submod_sigs:
                ambiguous.append({
                    "xmr": xmr_display,
                    "reason": (
                        f"no flat wire `{flat_name}` in top-module body AND "
                        f"signal `{sig}` not found in keep-hier submodule "
                        f"`{submod}` body of mapped.v "
                        f"(searched {len(submod_sigs)} submod-scope nets); "
                        f"chain trace={walk_info}."
                    ),
                })
                continue
            sig_range = submod_sigs[sig]
            rng_m = re.match(r"\[\s*(-?\d+)\s*:\s*(-?\d+)\s*\]", sig_range)
            if not is_indexed:
                # Scalar / vector wire that VCS can resolve hierarchically.
                # Emit a no-op rule so the substitution log captures that we
                # recognized + accepted this XMR (and so the auto-rules count
                # stays meaningful). Replacement == original.
                rules.append({
                    "regex": regex_nob,
                    "replacement": kept_path,
                    "_autogen": True, "_kind": "keep_hier_noop",
                    "_detail": (
                        f"{kept_path} resolves natively in "
                        f"keep-hier submod `{submod}` (range={sig_range!r}); "
                        f"no rewrite needed."
                        + (f" chain={walk_info}" if len(inner_chain) > 1 else "")
                    ),
                })
                continue
            # Indexed XMR: need element width to derive the slicing rule.
            # Owner = the submod that actually declares `sig` (the chain's
            # final node).
            owner = submod
            elem_w: int | None = inner_widths.get((owner, sig))
            if elem_w is None and not rng_m:
                ambiguous.append({
                    "xmr": xmr_display,
                    "reason": (
                        f"keep-hier lookup found submod `{submod}` signal "
                        f"`{sig}` (range={sig_range!r}) but could not infer "
                        f"element width from rtl_dut_files; "
                        f"owning_module={owner!r}."
                    ),
                })
                continue
            if not rng_m:
                # Non-literal range string — refuse to guess.
                ambiguous.append({
                    "xmr": xmr_display,
                    "reason": (
                        f"keep-hier signal `{submod}.{sig}` has non-literal "
                        f"range {sig_range!r}; cannot derive element count."
                    ),
                })
                continue
            msb, lsb = int(rng_m.group(1)), int(rng_m.group(2))
            total_w = abs(msb - lsb) + 1
            if elem_w is None:
                # If RTL didn't yield elem_w but the signal is declared as a
                # scalar / 1-bit vector in the submod, the XMR is over-indexed
                # and we refuse to guess. Otherwise we can't know.
                ambiguous.append({
                    "xmr": xmr_display,
                    "reason": (
                        f"keep-hier signal `{submod}.{sig}` exists "
                        f"(range={sig_range!r}, total_w={total_w}) but "
                        f"element width unknown from rtl_dut_files "
                        f"(owner={owner!r})."
                    ),
                })
                continue
            if total_w % elem_w != 0:
                ambiguous.append({
                    "xmr": xmr_display,
                    "reason": (
                        f"keep-hier signal `{submod}.{sig}` total width "
                        f"{total_w} not a multiple of inferred element "
                        f"width {elem_w}; refusing to guess slicing."
                    ),
                })
                continue
            n_elem = total_w // elem_w
            # MSB-first packing convention (same as flat path): mem[0]
            # occupies the top bits. Left-hand path stays hierarchical
            # because the wire is in the submod body.
            replacement = (f"{kept_path}"
                            f"[({n_elem - 1}-(\\1))*{elem_w} +: {elem_w}]")
            rules.append({
                "regex": regex_idx,
                "replacement": replacement,
                "_autogen": True, "_kind": "keep_hier_indexed_msb_first_slice",
                "_detail": (
                    f"{kept_path}[i] -> "
                    f"{kept_path}[(N-1-i)*W +: W] "
                    f"with N={n_elem}, W={elem_w} "
                    f"(submod={submod}, range={sig_range!r}, owner={owner})"
                    + (f", chain={walk_info}" if len(inner_chain) > 1 else "")
                ),
            })
            continue
        if not is_indexed:
            rules.append({
                "regex": regex_nob,
                "replacement": flat_path,
                "_autogen": True, "_kind": "scalar_rename",
                "_detail": f"{kept_path} -> {flat_path}",
            })
            continue
        # Indexed: need element width. For 1-inner chain, ask the RTL grep
        # which module declares `<inner_last>`; for deeper chains, the
        # netlist-side keep-hier walk would have caught these already (the
        # flat path is a flattened-netlist path, deep chains rarely show up
        # here), but the RTL lookup is kept as a fallback.
        owner = _grep_inner_module_for_inst(rtl_dut_files, inner_chain[-1])
        elem_w: int | None = None
        if owner is not None and (owner, sig) in inner_widths:
            elem_w = inner_widths[(owner, sig)]
        if elem_w is None:
            ambiguous.append({
                "xmr": xmr_display,
                "reason": (
                    "could not infer element width from rtl_dut_files; "
                    f"owning_module_guess={owner!r}, "
                    f"flat_wire_range={flat_wires.get(flat_name)!r}. "
                    "Provide an explicit workload.tb_substitutions rule."
                ),
            })
            continue
        # Compute total width N from flat_wires range "[<msb>:<lsb>]".
        rng = flat_wires.get(flat_name) or ""
        rng_m = re.match(r"\[\s*(-?\d+)\s*:\s*(-?\d+)\s*\]", rng)
        if not rng_m:
            ambiguous.append({
                "xmr": xmr_display,
                "reason": (
                    f"flat wire `{flat_name}` has non-literal range "
                    f"{rng!r}; cannot derive element count."
                ),
            })
            continue
        msb, lsb = int(rng_m.group(1)), int(rng_m.group(2))
        total_w = abs(msb - lsb) + 1
        if total_w % elem_w != 0:
            ambiguous.append({
                "xmr": xmr_display,
                "reason": (
                    f"flat wire width {total_w} not a multiple of inferred "
                    f"element width {elem_w}; refusing to guess slicing."
                ),
            })
            continue
        n_elem = total_w // elem_w
        # DC's standard ungroup-flatten convention is MSB-first packing
        # (mem[0] occupies the top bits). Emit that shape.
        replacement = (f"{flat_path}"
                        f"[({n_elem - 1}-(\\1))*{elem_w} +: {elem_w}]")
        rules.append({
            "regex": regex_idx,
            "replacement": replacement,
            "_autogen": True, "_kind": "indexed_msb_first_slice",
            "_detail": (f"{kept_path}[i] -> "
                         f"{flat_path}[(N-1-i)*W +: W] "
                         f"with N={n_elem}, W={elem_w} "
                         f"(owner={owner})"),
        })
    return rules, ambiguous


# ---------------------------------------------------------------------------
# Parameterized-macro post-uniquify alias resolution.
#
# When `set_dont_touch [get_designs <M>]` is applied to a parameterized macro
# (e.g. an SRAM model), DC uniquifies each instantiation into a
# parameter-suffixed clone (`sram_DEPTH128_D_WIDTH8_NUM_BANK16`, ...).
# The mapped.v emits each clone as an empty-bodied stub module (port list
# only) and rewrites every instance to the suffixed name. `macro_rtl_subs`
# only supplies the *generic* RTL body, so the suffixed instances bind to
# nothing → outputs are X across the netlist sim.
#
# This helper scans mapped.v for `<M>_<suffix>` stub declarations, parses the
# suffix into parameter overrides using the macro RTL's declared parameter
# list (in declaration order), and synthesizes a wrapper module per variant
# that instantiates `M` with the right `#(.PARAM(VALUE), ...)` overrides.
# The empty stubs are then removed from mapped.v via the existing strip path.
# ---------------------------------------------------------------------------


_MODULE_DECL_RE = re.compile(
    r"^\s*module\s+(?P<name>[A-Za-z_][A-Za-z0-9_$]*)\s*"
    r"\((?P<ports>[^;]*)\)\s*;",
    re.MULTILINE,
)


def _parse_macro_param_names(rtl_text: str, macro_module: str) -> list[str]:
    """Read the macro module's `module <M> #( parameter <NAME>... );` header
    and return the declared parameter names in source order.

    Skips `localparam` (DC doesn't carry them in the uniquify suffix) and
    only honors `parameter int|integer|bit|...` declarations. We don't try
    to parse defaults — they'd be ignored anyway, since the wrapper passes
    explicit overrides.
    """
    # Locate `module <M> #( ... )(`. We find the first `module <M>` then
    # look for an optional `#(...)` block.
    m = re.search(rf"\bmodule\s+{re.escape(macro_module)}\b", rtl_text)
    if not m:
        return []
    j = m.end()
    n = len(rtl_text)
    while j < n and rtl_text[j].isspace():
        j += 1
    if j >= n or rtl_text[j] != "#":
        return []  # no parameter block → no overrides expressible in suffix
    j += 1
    while j < n and rtl_text[j].isspace():
        j += 1
    if j >= n or rtl_text[j] != "(":
        return []
    # Find matching close paren (paren-balanced — defaults like $clog2(X) are
    # legal here too).
    depth = 1
    open_i = j
    j += 1
    while j < n and depth > 0:
        if rtl_text[j] == "(":
            depth += 1
        elif rtl_text[j] == ")":
            depth -= 1
            if depth == 0:
                break
        j += 1
    block = rtl_text[open_i + 1:j]
    # Extract `parameter` declarations only (skip localparam, comments, etc.).
    # The declaration pattern: `parameter [type] NAME = ...`.
    names: list[str] = []
    for pm in re.finditer(
        r"\bparameter\b(?!\s*(?:int|integer|bit|logic)?\s+\w+\s*=\s*localparam)"
        r"\s+(?:(?:int\s+unsigned|int|integer|bit|logic|real)\s+)?"
        r"(?P<name>[A-Za-z_][A-Za-z0-9_$]*)",
        block,
    ):
        nm = pm.group("name")
        if nm not in names:
            names.append(nm)
    return names


def _parse_uniquify_suffix(suffix: str, param_names: list[str]
                            ) -> list[tuple[str, str]] | None:
    """Greedy-decode a DC uniquify suffix like ``DEPTH128_D_WIDTH8_NUM_BANK16``
    into ``[("DEPTH","128"),("D_WIDTH","8"),("NUM_BANK","16")]``, using the
    declared parameter list (in module-source order) as the alphabet.

    Returns ``None`` if the suffix cannot be decoded (we surface that as an
    ambiguity / orphan variant to the caller rather than guess wrong).
    """
    if not param_names:
        return None
    s = suffix
    out: list[tuple[str, str]] = []
    pi = 0  # next param to expect
    # Allow leading '_' between groups; DC's pattern is
    # ``<NAME1><VALUE1>_<NAME2><VALUE2>_...``.
    while s and pi < len(param_names):
        # Strip a separator underscore if we already consumed at least one.
        if out and s.startswith("_"):
            s = s[1:]
        nm = param_names[pi]
        if not s.startswith(nm):
            # Wrong order or unsupported param; bail.
            return None
        s = s[len(nm):]
        # Decimal value follows.
        v_match = re.match(r"-?\d+", s)
        if not v_match:
            return None
        out.append((nm, v_match.group(0)))
        s = s[v_match.end():]
        pi += 1
    if s:  # leftover characters → didn't fully consume
        return None
    return out


def _parse_mapped_module_ports(mapped_text: str, module_name: str
                                ) -> tuple[list[str], dict[str, str]] | None:
    """Pull a stub module's port list (in order) and direction/width
    declarations out of `mapped_text`. Returns (ordered_port_names,
    {port_name: "<dir> [<msb>:<lsb>]"} or None if not found).

    The stub format DC emits is::

        module M ( a, b, c );
          input [W-1:0] a;
          input b, c;
        endmodule

    A port can appear with or without a packed range, and DC sometimes
    groups multiple same-direction scalars on one line (`input clk, wr_en;`).
    """
    pattern = re.compile(
        rf"\bmodule\s+{re.escape(module_name)}\s*"
        r"\((?P<ports>[^;]*)\)\s*;\s*(?P<body>.*?)\bendmodule\b",
        re.DOTALL,
    )
    m = pattern.search(mapped_text)
    if not m:
        return None
    ports_str = m.group("ports")
    body = m.group("body")
    ordered = [p.strip() for p in re.split(r"\s*,\s*", ports_str) if p.strip()]

    decl_re = re.compile(
        r"\b(?P<dir>input|output|inout)\b"
        r"\s*(?P<range>\[\s*[^\]]+?\s*\])?"
        r"\s+(?P<names>[A-Za-z_][A-Za-z0-9_$,\s]*)"
        r"\s*;",
    )
    decls: dict[str, str] = {}
    for d in decl_re.finditer(body):
        rng = d.group("range") or ""
        rng_str = (rng + " ") if rng else ""
        for nm in (n.strip() for n in d.group("names").split(",")):
            if nm:
                decls[nm] = f"{d.group('dir')} {rng_str}".rstrip()
    return ordered, decls


def _emit_uniquify_wrapper_sv(
    macro_module: str,
    variant_name: str,
    ports_in_order: list[str],
    port_decls: dict[str, str],
    overrides: list[tuple[str, str]],
) -> str:
    """Emit a per-variant wrapper module:

        module <macro>_<suffix> (... ports ...);
          <input/output decls copied from mapped.v stub>
          <macro> #( .PARAM(VALUE), ... ) u_inner ( .port(port), ... );
        endmodule
    """
    header = f"module {variant_name} ({', '.join(ports_in_order)});\n"
    decl_lines: list[str] = []
    for p in ports_in_order:
        d = port_decls.get(p, "input")
        decl_lines.append(f"  {d} {p};")
    if overrides:
        param_str = ", ".join(f".{nm}({val})" for nm, val in overrides)
        param_block = f" #({param_str})"
    else:
        param_block = ""
    conn_str = ", ".join(f".{p}({p})" for p in ports_in_order)
    inst = f"  {macro_module}{param_block} u_inner ({conn_str});\n"
    return header + "\n".join(decl_lines) + "\n" + inst + "endmodule\n"


def _resolve_uniquified_macro_variants(
    mapped_text: str,
    macro_subs: list[dict],
    arts: Path,
    log,
) -> tuple[list[Path], list[str], list[dict], list[dict]]:
    """For each `macro_rtl_subs` entry, find every `<M>_<suffix>` stub module
    in `mapped_text`, synthesize a wrapper module per variant, and emit them
    to a single .sv file under `arts/`.

    Returns (wrapper_paths, strip_module_names, resolved_log, orphan_log).
    Caller appends `wrapper_paths` to the netlist filelist and feeds
    `strip_module_names` into the existing strip-from-mapped pass.

    `resolved_log` shape: [{generic, variants: [{name, params}]}]
    `orphan_log`   shape: [{variant, generic_guess}]
    """
    if not macro_subs:
        return [], [], [], []

    # Index module-name set in mapped.v (pull directly from the regex above).
    declared = {m.group("name") for m in _MODULE_DECL_RE.finditer(mapped_text)}

    # Build a generic-name → param_names map; we need this for suffix decode.
    rtl_text_by_macro: dict[str, str] = {}
    for ms in macro_subs:
        macro = ms.get("macro_module")
        if macro and Path(ms["rtl_file"]).is_file():
            rtl_text_by_macro[macro] = Path(ms["rtl_file"]).read_text()
    param_names_by_macro: dict[str, list[str]] = {
        m: _parse_macro_param_names(t, m) for m, t in rtl_text_by_macro.items()
    }

    resolved_log: list[dict] = []
    orphan_log: list[dict] = []
    strip_names: list[str] = []
    wrapper_blocks: list[str] = []
    seen_variants: set[str] = set()

    macro_names = sorted(rtl_text_by_macro.keys(), key=lambda s: -len(s))

    for macro in macro_names:
        prefix = macro + "_"
        variants = sorted(n for n in declared
                          if n.startswith(prefix) and n != macro)
        if not variants:
            continue
        params_in_order = param_names_by_macro.get(macro, [])
        per_variant: list[dict] = []
        for v in variants:
            if v in seen_variants:
                continue
            seen_variants.add(v)
            suffix = v[len(prefix):]
            decoded = _parse_uniquify_suffix(suffix, params_in_order)
            if decoded is None:
                orphan_log.append({
                    "variant": v,
                    "generic_guess": macro,
                    "reason": (
                        "could not decode suffix into the macro's declared "
                        f"parameter order {params_in_order!r}"
                    ),
                })
                continue
            ports_info = _parse_mapped_module_ports(mapped_text, v)
            if ports_info is None:
                orphan_log.append({
                    "variant": v, "generic_guess": macro,
                    "reason": "stub module declaration not found in mapped.v",
                })
                continue
            ports_in_order, port_decls = ports_info
            wrap_sv = _emit_uniquify_wrapper_sv(
                macro, v, ports_in_order, port_decls, decoded,
            )
            wrapper_blocks.append(wrap_sv)
            strip_names.append(v)
            per_variant.append({"name": v,
                                "params": [{"name": nm, "value": val}
                                            for nm, val in decoded]})
            log(f"  uniquified-macro alias: {macro} -> {v} via "
                + ", ".join(f"{nm}={val}" for nm, val in decoded))
        if per_variant:
            resolved_log.append({"generic": macro, "variants": per_variant})

    # Detect orphans: any module whose name has prefix `<M>_` for some unbound
    # M (where M is NOT in macro_subs but might be in another file).
    declared_macros = set(rtl_text_by_macro.keys())
    bound_or_alias = set(declared_macros) | set(strip_names)
    for nm in declared:
        if nm in bound_or_alias:
            continue
        # Look for prefix `<X>_` where X seems to be a parameterized macro
        # but X is not bound. We surface only those whose generic body is
        # not declared anywhere — bare module name is the user's signal.
        # (Parameterized macro variants always have an underscore in the
        # suffix; we keep this conservative — only flag if X happens to be
        # the prefix of an existing rtl_subs entry but not a covered case.)
        # The heuristic above already covers covered macros; nothing else
        # to add here without producing false positives.
        pass

    if not wrapper_blocks:
        return [], [], resolved_log, orphan_log

    wrappers_path = arts / "macro_uniquify_wrappers.sv"
    header = (
        "// ============================================================\n"
        "// Auto-generated by setfi.workload.adapt_tb (uniquify aliases).\n"
        "// One wrapper per <macro>_<suffix> uniquified clone DC emitted\n"
        "// for parameterized macros under set_dont_touch [get_designs ...].\n"
        "// Each wrapper instantiates the generic macro with the parameter\n"
        "// overrides decoded from the suffix, restoring functional binding\n"
        "// in the netlist sim. Empty stubs from mapped.v are stripped via\n"
        "// netlist_strip_modules.\n"
        "// ============================================================\n"
        "`default_nettype none\n"
        "`timescale 1ns/1ps\n\n"
    )
    wrappers_path.write_text(header + "\n".join(wrapper_blocks))
    return [wrappers_path], strip_names, resolved_log, orphan_log


def _strip_modules_from_mapped(mapped_text: str,
                                names: list[str]) -> tuple[str, dict[str, int]]:
    """Strip every ``^module <name>(...)`` ... ``^endmodule`` block whose
    module name is in ``names``. Returns (text, {name: n_stripped}).

    Used for modules that cannot be simulated from the netlist (e.g. left
    unmapped by synthesis); the caller provides their replacements via
    ``netlist_substitute_files``.
    """
    if not names:
        return mapped_text, {}
    out_lines: list[str] = []
    counts: dict[str, int] = {n: 0 for n in names}
    in_strip = False
    cur_name = ""
    mod_re = re.compile(r"^\s*module\s+(\w+)\s*[(\s;]")
    for line in mapped_text.splitlines(keepends=True):
        if not in_strip:
            m = mod_re.match(line)
            if m and m.group(1) in counts:
                in_strip = True
                cur_name = m.group(1)
                counts[cur_name] += 1
                continue
            out_lines.append(line)
        else:
            if line.lstrip().startswith("endmodule"):
                in_strip = False
                cur_name = ""
            # else: drop the line
    return "".join(out_lines), counts



def _write_filelist(path: Path, files: Sequence[Path]) -> None:
    path.write_text("\n".join([str(f) for f in files] + [""]))


@dataclass
class AdaptResult:
    """What the netlist simulation compiles, in order:
    ``cell_models + [sim_netlist] + extra_sources + tb_files``."""
    tb_files: List[Path]
    sim_netlist: Path
    extra_sources: List[Path]
    cell_models: List[Path]
    net_filelist: Path
    netlist_was_stripped: bool = False
    report: Dict = field(default_factory=dict)


def adapt_testbench(
    *,
    tb_files: Sequence,
    netlist,
    cell_models: Sequence,
    top_module: str,
    tb_top_module: str,
    out_dir,
    rtl_dut_files: Sequence = (),
    tb_dut_inst: str = "dut",
    hier_substitutions: Sequence[dict] = (),
    auto_derive_hier_substitutions: bool = False,
    macro_rtl_subs: Sequence[dict] = (),
    netlist_strip_modules: Sequence[str] = (),
    netlist_substitute_files: Sequence = (),
    log: Optional[Callable[[str], None]] = None,
) -> AdaptResult:
    """Prepare the testbench and file list for the gate-level simulation.

    Writes into ``out_dir``: the adapted testbench (``tb_netlist.sv`` for a
    single file, ``tb_netlist_<k>_<name>`` for several) when any adaptation is
    configured, the stripped netlist copy ``mapped_adapted.v`` and
    ``macro_uniquify_wrappers.sv`` when needed, ``sim_net_filelist.f``,
    ``result.json`` and ``run.log``.  Raises :class:`WorkloadError` on failure
    (``result.json`` then carries the error).
    """
    started = time.monotonic()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arts = out_dir
    log_lines: List[str] = []

    def _log(m: str) -> None:
        log_lines.append(m)
        if log is not None:
            log(m)

    report: Dict = {"tool": TOOL_NAME, "version": TOOL_VERSION, "status": "ok", "error": None,
                    "inputs": None, "outputs": {}, "metrics": {},
                    "meta": {"started_iso": _now_iso(), "finished_iso": None, "duration_s": None}}
    try:
        tb_paths = [Path(p).resolve() for p in tb_files]
        if not tb_paths:
            raise WorkloadError("config_invalid", "no testbench file given")
        for p in tb_paths:
            if not p.is_file():
                raise WorkloadError("config_invalid", f"testbench file missing: {p}")
        rtl_paths = _resolve_files(list(rtl_dut_files), "rtl_dut_files")
        mapped_v = Path(netlist).resolve()
        if not mapped_v.is_file():
            raise WorkloadError("config_invalid", f"netlist missing: {mapped_v}")
        tech_models = _resolve_files(list(cell_models), "cell_models", "tech_lib_missing")

        macro_subs = list(macro_rtl_subs or [])
        for ms in macro_subs:
            if not isinstance(ms, dict) or "rtl_file" not in ms:
                raise WorkloadError("macro_unhandled",
                                    f"workload.macro_rtl_substitutions entry needs rtl_file: {ms}")
            if not Path(ms["rtl_file"]).resolve().is_file():
                raise WorkloadError("macro_unhandled",
                                    f"workload.macro_rtl_substitutions: rtl_file not found: {ms['rtl_file']}")

        # Parameterized black-box macros: alias uniquified stub clones to the
        # generic RTL before the file list is fixed.
        mapped_text_in = mapped_v.read_text()
        wrapper_paths, uniq_strip, uniq_resolved, uniq_orphans = \
            _resolve_uniquified_macro_variants(mapped_text_in, macro_subs, arts, _log)
        if uniq_orphans:
            raise WorkloadError(
                "macro_unhandled",
                "uniquified_macro_unbound: variants present in the netlist "
                f"could not be aliased to a generic body -- {uniq_orphans!r}",
            )

        strip_mods = list(netlist_strip_modules or [])
        sub_files = list(netlist_substitute_files or [])
        strip_mods.extend(uniq_strip)
        if strip_mods:
            stripped_text, strip_counts = _strip_modules_from_mapped(mapped_text_in, list(strip_mods))
            mapped_for_sim = arts / "mapped_adapted.v"
            mapped_for_sim.write_text(stripped_text)
            _log(f"  stripped modules from netlist: {strip_counts}")
            missing = [n for n, c in strip_counts.items() if c == 0]
            if missing:
                raise WorkloadError("config_invalid",
                                    f"netlist_strip_modules listed modules NOT present "
                                    f"in the netlist: {missing}")
        else:
            mapped_for_sim = mapped_v
            strip_counts = {}

        sub_paths = _resolve_files(sub_files, "netlist_substitute_files") if sub_files else []
        macro_paths = [Path(ms["rtl_file"]).resolve() for ms in macro_subs]
        wrapper_paths = [Path(wp).resolve() for wp in wrapper_paths]

        # Testbench text edits: substitutions (manual, then auto-derived).
        subs = list(hier_substitutions or [])
        texts = [p.read_text() for p in tb_paths]

        auto_rules: List[dict] = []
        if auto_derive_hier_substitutions:
            probe_text = "\n".join(texts)
            if subs:
                probe_text, _ = _apply_substitutions(probe_text, subs)
            auto_rules, ambiguous = _auto_derive_hier_substitutions(
                probe_text, mapped_text_in, top_module, rtl_paths, tb_dut_inst,
            )
            if ambiguous:
                raise WorkloadError(
                    "tb_unparseable",
                    "hier_substitution_ambiguous: auto-derive could not resolve some "
                    "testbench XMRs unambiguously; supply explicit workload.tb_substitutions "
                    f"rules for: {ambiguous!r}",
                )
            subs.extend(auto_rules)
            _log(f"  auto-derived {len(auto_rules)} hier substitution rule(s)")
            for ar in auto_rules:
                _log(f"    auto: [{ar.get('_kind')}] {ar.get('_detail')}")

        applied_log: List[dict] = []
        if subs:
            new_texts = []
            per_file_logs = []
            for t in texts:
                t2, alog = _apply_substitutions(t, subs)
                new_texts.append(t2)
                per_file_logs.append(alog)
            texts = new_texts
            applied_log = [dict(per_file_logs[0][i], n_subs=sum(fl[i]["n_subs"] for fl in per_file_logs))
                           for i in range(len(subs))]

        tb_for_net: List[Path]
        if subs:
            tb_for_net = []
            for k, (p, t) in enumerate(zip(tb_paths, texts)):
                name = "tb_netlist.sv" if len(tb_paths) == 1 else f"tb_netlist_{k}_{p.name}"
                adapted = arts / name
                adapted.write_text(t)
                tb_for_net.append(adapted)
            _log(f"  applied {len(applied_log)} testbench substitution(s)")
            for a in applied_log:
                _log(f"    sub: {a['regex']!r} -> {a['replacement']!r}: {a['n_subs']} match(es)")
        else:
            tb_for_net = list(tb_paths)
            _log("  no testbench adaptations; using the original testbench verbatim")

        netlist_files = ([mapped_for_sim] + list(sub_paths) + list(tech_models)
                         + macro_paths + wrapper_paths + tb_for_net)
        net_filelist = arts / "sim_net_filelist.f"
        _write_filelist(net_filelist, netlist_files)

        # Sources the simulation needs between the netlist and the testbench.
        # Only when a module was stripped or a uniquified macro aliased: with
        # neither, a macro's generic RTL would duplicate the netlist's module.
        extra_sources: List[Path] = []
        if strip_mods or uniq_resolved:
            excl = {str(p) for p in tech_models} | {str(mapped_for_sim)} | {str(p) for p in tb_for_net}
            for p in list(sub_paths) + macro_paths + wrapper_paths:
                if str(p) not in excl:
                    extra_sources.append(p)
            sim_netlist = mapped_for_sim
        else:
            sim_netlist = mapped_v

        report["inputs"] = {
            "tb_files": [str(p) for p in tb_paths],
            "rtl_dut_files": [str(p) for p in rtl_paths],
            "netlist": str(mapped_v),
            "cell_models": [str(p) for p in tech_models],
            "macro_rtl_substitutions": macro_subs,
            "top_module": top_module,
            "tb_top_module": tb_top_module,
            "tb_substitutions": subs,
        }
        report["outputs"] = {
            "net_filelist": str(net_filelist),
            "sim_netlist": str(sim_netlist),
            "netlist_was_stripped": bool(strip_mods),
            "tb_files": [str(p) for p in tb_for_net],
            "tb_was_modified": tb_for_net != list(tb_paths),
            "extra_sources": [str(p) for p in extra_sources],
            "macro_variants": uniq_resolved,
        }
        report["metrics"] = {
            "n_netlist_files": len(netlist_files),
            "n_macro_rtl_substitutions": len(macro_subs),
            "n_tb_substitutions_applied": sum(a["n_subs"] for a in applied_log),
            "n_tb_substitution_rules": len(applied_log),
            "tb_substitution_log": applied_log,
            "n_strip_modules": len(strip_mods),
            "strip_module_counts": strip_counts,
            "n_substitute_files": len(sub_paths),
            "n_macro_variant_wrapper_files": len(wrapper_paths),
            "n_macro_variants": sum(len(g["variants"]) for g in uniq_resolved),
            "n_auto_derived_substitutions": len(auto_rules),
            "auto_derived_substitutions": [{k: v for k, v in r.items()
                                         if k in ("regex", "replacement", "_kind", "_detail")}
                                        for r in auto_rules],
        }
        _log(f"  net filelist: {net_filelist} ({len(netlist_files)} files)")
        result = AdaptResult(tb_files=tb_for_net, sim_netlist=sim_netlist,
                             extra_sources=extra_sources, cell_models=tech_models,
                             net_filelist=net_filelist,
                             netlist_was_stripped=bool(strip_mods), report=report)
    except WorkloadError as e:
        report["status"] = "error"
        report["error"] = {"kind": e.kind, "message": e.message}
        _log(f"ERROR [{e.kind}]: {e.message}")
        raise
    finally:
        report["meta"]["finished_iso"] = _now_iso()
        report["meta"]["duration_s"] = round(time.monotonic() - started, 4)
        (out_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        (out_dir / "run.log").write_text("\n".join(log_lines) + "\n")
    return result
