#!/usr/bin/env python3
"""Evaluate the published Gaz et al. (RA-L 2019) Panda model functions as an independent numeric oracle.

The MATLAB files (get_GravityVector.m, get_MassMatrix.m) are symbolic-toolbox output: straight-line
assignments `name = expr;` using q1..q7, t<k>, kinematic constants, cos/sin, numeric literals and
arithmetic, ending in a column vector / stacked matrix. They are UNTRUSTED downloaded content: this
translator accepts only that grammar (identifier allowlist, no function other than cos/sin, no
builtins in the evaluation namespace) and never executes anything else. Diagnostic only.
The files are GPL-3.0 (github.com/marcocognetti/FrankaEmikaPandaDynModel); they are NOT copied into the
repository - only read from a local download directory given on the command line.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

IDENT = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
ALLOWED_FUN = {"cos", "sin"}


def _translate(expr: str) -> str:
    e = expr.replace(".*", "*").replace("./", "/").replace(".^", "**").replace("^", "**")
    return e


def load(path: Path):
    """Returns a function f(q: list[float]) -> list[float] (flattened output in MATLAB order)."""
    text = Path(path).read_text()
    body = text.split("function", 1)[1].split("\n", 1)[1]
    # join continuation lines
    body = body.replace("...\n", " ")
    stmts = []
    for line in body.splitlines():
        line = line.split("%", 1)[0].strip()
        if not line:
            continue
        if not line.endswith(";"):
            raise ValueError("unexpected statement form: " + line[:60])
        name, expr = line[:-1].split("=", 1)
        stmts.append((name.strip(), expr.strip()))
    known = {f"q{i}" for i in range(1, 8)} | {"q"}
    out_name = None
    code = []
    for name, expr in stmts:
        if re.fullmatch(r"q\d = q\(\d\)", f"{name} = {expr}"):
            continue
        if expr.startswith("["):  # final vector / stacked matrix
            inner = expr.strip("[]")
            items = [x.strip() for row in inner.split(";") for x in row.split(",") if x.strip()]
            items = [re.sub(r"(\w+)\((\d+)\)", lambda m: f"{m.group(1)}[{int(m.group(2)) - 1}]", x) for x in items]
            for it in items:
                for tok in IDENT.findall(it):
                    if tok not in known and tok not in ALLOWED_FUN and tok != "e":
                        raise ValueError("unknown identifier in output: " + tok)
            out_name = name
            code.append((name, "[" + ",".join(_translate(x) for x in items) + "]"))
            known.add(name)
            continue
        py = _translate(expr)
        py = re.sub(r"(\w+)\((\d+)\)", lambda m: f"{m.group(1)}[{int(m.group(2)) - 1}]"
                    if m.group(1) not in ALLOWED_FUN else m.group(0), py)
        for tok in IDENT.findall(py):
            if tok in ALLOWED_FUN or tok in known or re.fullmatch(r"e", tok):
                continue
            raise ValueError(f"identifier {tok!r} not allowed")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name):
            raise ValueError("bad assignment target")
        code.append((name, py))
        known.add(name)
    compiled = [(n, compile(e, "<gaz>", "eval")) for n, e in code]

    def f(q):
        env = {"__builtins__": {}, "cos": math.cos, "sin": math.sin, "q": list(q)}
        for i in range(7):
            env[f"q{i + 1}"] = q[i]
        for n, c in compiled:
            env[n] = eval(c, env)  # noqa: S307 - restricted grammar, no builtins
        return env[out_name]
    return f
