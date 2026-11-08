"""Translate EvoApproxLib gate-level C models into vectorized numpy callables.

The library ships several C code styles (gate signals, wa/wb arrays, dout_*
declarations); all are straight-line bitwise code, so a statement-level
rewrite into Python is exact.
"""
import re
from pathlib import Path

import numpy as np

EVO_ROOT = (Path(__file__).resolve().parents[1] / "results" / "research_20261022"
            / "primary_sources" / "evoapproxlib_full_v2022" / "adders")

_SIG = re.compile(r"\b(?:u?int\d+_t|int)\s+(add\w+)\s*\(([^)]*)\)\s*\{(.*)\}", re.S)
_TYPE = re.compile(r"^(?:(?:const|unsigned|static|u?int\d+_t|int)\s+)+")


def _pyexpr(e):
    e = re.sub(r"\((?:u?int\d*_t|int)\)", "", e)
    e = re.sub(r"(0x[0-9A-Fa-f]+|\d+)[uUlL]+\b", r"\1", e)
    e = e.replace("0xFFFFFFFFFFFFFFFF", "(-1)")
    e = re.sub(r"!\s*([\w\[\]]+)", r"((\1) == 0) * 1", e)  # C logical not
    return e


def load_cell(path):
    """Return (name, f) where f(a, b) evaluates the C model on int64 arrays (a = first operand)."""
    src = Path(path).read_text()
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    src = re.sub(r"//[^\n]*", "", src)
    m = _SIG.search(src)
    if m is None:
        raise ValueError(f"unrecognized C model: {path}")
    name, params, body = m.groups()
    pnames = [p.split()[-1] for p in params.split(",")]
    lines, ret = [], None
    for raw in body.split(";"):
        stmt = " ".join(raw.split())
        if not stmt:
            continue
        if stmt.startswith("return"):
            ret = stmt.split()[1]
            continue
        is_decl = bool(_TYPE.match(stmt))
        stmt = _TYPE.sub("", stmt)
        if is_decl:
            for part in stmt.split(","):
                part = part.strip()
                arr = re.fullmatch(r"(\w+)\[(\d+)\]", part)
                if arr:
                    lines.append(f"{arr.group(1)} = [0] * {arr.group(2)}")
                elif "=" in part:
                    lhs, rhs = part.split("=", 1)
                    lines.append(f"{lhs.strip()} = {_pyexpr(rhs)}")
                else:
                    lines.append(f"{part} = 0")
            continue
        op = re.match(r"([\w\[\]]+)\s*([|&^]?=)\s*(.*)", stmt)
        if op is None:
            raise ValueError(f"cannot parse statement {stmt!r} in {path}")
        lhs, eq, rhs = op.groups()
        rhs = _pyexpr(rhs)
        lines.append(f"{lhs} = {lhs} {eq[0]} ({rhs})" if eq != "=" else f"{lhs} = {rhs}")
    code = f"def _f({pnames[0]}, {pnames[1]}):\n" + "".join(f"    {s}\n" for s in lines) \
        + f"    return {ret}\n"
    env = {}
    exec(compile(code, f"<{name}>", "exec"), env)
    f = env["_f"]

    def call(a, b):
        return np.asarray(f(np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)),
                          dtype=np.int64)

    return name, call


def cells(folder):
    """Yield unique (name, f, metadata) for every C model under folder."""
    seen = set()
    for p in sorted(Path(folder).rglob("*.c")):
        if p.stem in seen:
            continue
        seen.add(p.stem)
        name, f = load_cell(p)
        meta = dict(re.findall(r"//\s*([A-Z0-9_%]+)\s*=\s*([-0-9.eE]+)", p.read_text()))
        yield name, f, meta
