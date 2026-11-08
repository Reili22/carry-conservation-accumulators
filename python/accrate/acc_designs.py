"""Accumulator RTL generators and bit-exact Python reference models.

Every design is a W-bit accumulator T <= F(T, in) with an IW-bit
operand.  Families:
  exact          T + in
  trunc_t        exact accumulation of in with its low t bits dropped
  round_t        exact accumulation of in rounded half-up at bit t
  tplus_t        exact accumulation of in truncated at bit t plus one LSB(t) per operation
  evo_<cell>     EvoApprox cell (8/12/16-bit, residue width k<=8) as low part, exact upper part
  loa_k, ...     classic lower-part approximate adders (see classic_lpaa)
Registers below the kept precision are not instantiated, so their cost is real.
Every design has a synchronous clear (clr) to zero, as in a MAC unit.
"""
import json
import os
import re
from pathlib import Path

import numpy as np

from evo_c2py import EVO_ROOT, load_cell

W, IW = 24, 16
MASK = (1 << W) - 1


# ACC_SIGNED=1 sign-extends the 16-bit operand (two's-complement MAC products);
# otherwise it is zero-extended.  The accumulator circuit is otherwise identical.
SIGNED = os.environ.get("ACC_SIGNED", "0") == "1"
_CERT = Path(__file__).with_name("carry_census.json")


def evo_width(cell):
    return int(re.match(r"add(\d+)u_", cell).group(1))


def evo_source(cell):
    hits = [p for p in (EVO_ROOT / f"{evo_width(cell)}_unsigned").rglob(f"{cell}.v")]
    return hits[0]


def evo_support(cell):
    """Smallest residue width k of a cell.

    8-bit cells: exhaustive over all 2^16 operand pairs.  Wider cells: the
    SAT-certified width recorded in carry_census.json (k <= 8 only).
    """
    w = evo_width(cell)
    if w != 8:
        return {r["cell"]: r["k"] for r in json.loads(_CERT.read_text())}[cell]
    _, f = load_cell(evo_source(cell).with_suffix(".c"))
    a, b = np.meshgrid(np.arange(256), np.arange(256), indexing="ij")
    E = f(a, b) - a - b
    for k in range(9):
        q = 1 << k
        T = E.reshape(256 // q, q, 256 // q, q)
        if np.all(T == T[0:1, :, 0:1, :]):
            return k
    raise AssertionError(cell)


def _ext(lo):
    """Operand bits [IW-1:lo] extended to W-lo bits (zero or sign extension)."""
    fill = f"{{{W-IW}{{in[{IW-1}]}}}}" if SIGNED else f"{W-IW}'d0"
    return f"{{{fill}, in[{IW-1}:{lo}]}}"


def rtl(design):
    """Return (top_name, verilog_text, extra_source_paths)."""
    kind, _, arg = design.partition("_")
    top = f"acc_{design}"
    head = f"`default_nettype none\nmodule {top}(input clk, input clr, input [{IW-1}:0] in, output [{W-1}:0] out);\n"
    if kind == "exact" or (kind in ("trunc", "round") and int(arg) == 0):
        body = (f"  reg [{W-1}:0] T;\n  always @(posedge clk) T <= clr ? 0 : T + {_ext(0)};\n  assign out = T;\n")
        return top, head + body + "endmodule\n", []
    if kind in ("trunc", "round", "tplus"):
        t = int(arg)
        rnd = {"round": f" + in[{t-1}]", "tplus": " + 1'b1", "trunc": ""}[kind]
        body = (f"  reg [{W-1}:{t}] T;\n"
                f"  always @(posedge clk) T <= clr ? 0 : T + {_ext(t)}{rnd};\n"
                f"  assign out = {{T, {t}'d0}};\n")
        return top, head + body + "endmodule\n", []
    if kind == "evo":
        cell = arg
        w = evo_width(cell)
        k = evo_support(cell)
        if k == 0:
            k = 8  # exact library cell: the residue property holds for every k >= 0
        if k == w:
            low_a, low_b = f"T[{w-1}:0]", f"in[{w-1}:0]"
        else:
            # Residue property (error depends only on the low k bits): with zeroed high
            # bits the cell outputs z = o[k-1:0] and its carry c = o[k] (c is 0 or 1);
            # bits >= k then use one exact adder, as in the rounding/truncation designs.
            low_a, low_b = f"{{{w-k}'d0, T[{k-1}:0]}}", f"{{{w-k}'d0, in[{k-1}:0]}}"
        body = (f"  reg [{W-1}:0] T;\n  wire [{w}:0] o;\n"
                f"  {cell} u_low(.A({low_a}), .B({low_b}), .O(o));\n"
                f"  always @(posedge clk) T <= clr ? 0 : {{T[{W-1}:{k}] + {_ext(k)} + o[{k}], o[{k-1}:0]}};\n"
                f"  assign out = T;\n")
        return top, head + body + "endmodule\n", [evo_source(cell)]
    if kind in CLASSIC:
        k = int(arg)
        low_v, _ = CLASSIC[kind]
        body = (f"  reg [{W-1}:0] T;\n  wire [{k-1}:0] x = T[{k-1}:0], u = in[{k-1}:0];\n"
                f"  wire [{k-1}:0] z; wire c;\n{low_v(k)}"
                f"  always @(posedge clk) T <= clr ? 0 : {{T[{W-1}:{k}] + {_ext(k)} + c, z}};\n"
                f"  assign out = T;\n")
        return top, head + body + "endmodule\n", []
    raise ValueError(design)


def operand24(U16):
    """The W-bit operand the RTL adds for 16-bit input patterns U16."""
    U16 = np.asarray(U16, np.int64) & 0xFFFF
    if SIGNED:
        U16 = np.where(U16 >= 0x8000, U16 - 0x10000, U16)
    return U16 & MASK


# ---- classic lower-part approximate adders: (verilog(k), python(x,u,k)->(z,c)) ----

def _loa_v(k):
    return f"  assign z = x | u;\n  assign c = x[{k-1}] & u[{k-1}];\n"


def _loa_py(x, u, k):
    return x | u, (x >> (k - 1)) & (u >> (k - 1)) & 1


def _loawa_v(k):
    return "  assign z = x | u;\n  assign c = 1'b0;\n"


def _loawa_py(x, u, k):
    return x | u, np.zeros_like(x)


def _eta1_v(k):
    # ETA-I: from the MSB down, XOR until the first position with both bits 1; that bit and all below are 1.
    lines = [f"  wire [{k}:0] hit;\n  assign hit[{k}] = 1'b0;\n"]
    for i in range(k - 1, -1, -1):
        lines.append(f"  assign hit[{i}] = hit[{i+1}] | (x[{i}] & u[{i}]);\n")
        lines.append(f"  assign z[{i}] = hit[{i}] ? 1'b1 : (x[{i}] ^ u[{i}]);\n")
    lines.append("  assign c = 1'b0;\n")
    return "".join(lines)


def _eta1_py(x, u, k):
    z = np.zeros_like(x)
    hit = np.zeros_like(x)
    for i in range(k - 1, -1, -1):
        xi, ui = (x >> i) & 1, (u >> i) & 1
        hit = hit | (xi & ui)
        z |= np.where(hit == 1, 1, xi ^ ui) << i
    return z, np.zeros_like(x)


def _ama5_v(k):
    # Gupta et al. approximation 5 per bit (S = B, Cout = A), rippled: carry-out = x[k-1].
    return f"  assign z = u;\n  assign c = x[{k-1}];\n"


def _ama5_py(x, u, k):
    return u.copy(), (x >> (k - 1)) & 1


def _ama2_v(k):
    # Gupta approximation 2: exact carry ripple, S = ~Cout.
    lines = [f"  wire [{k}:0] cc;\n  assign cc[0] = 1'b0;\n"]
    for i in range(k):
        lines.append(f"  assign cc[{i+1}] = (x[{i}] & u[{i}]) | (cc[{i}] & (x[{i}] | u[{i}]));\n")
        lines.append(f"  assign z[{i}] = ~cc[{i+1}];\n")
    lines.append(f"  assign c = cc[{k}];\n")
    return "".join(lines)


def _ama2_py(x, u, k):
    cc = np.zeros_like(x)
    z = np.zeros_like(x)
    for i in range(k):
        xi, ui = (x >> i) & 1, (u >> i) & 1
        cc = (xi & ui) | (cc & (xi | ui))
        z |= (1 - cc) << i
    return z, cc


CLASSIC = {"loa": (_loa_v, _loa_py), "loawa": (_loawa_v, _loawa_py), "eta1": (_eta1_v, _eta1_py),
           "ama5": (_ama5_v, _ama5_py), "ama2": (_ama2_v, _ama2_py)}


def step_fn(design):
    """Return f(T, U) -> next T for numpy int64 arrays (bit-exact to rtl())."""
    kind, _, arg = design.partition("_")
    if kind == "exact":
        return lambda T, U: (T + U) & MASK
    if kind in ("trunc", "round", "tplus"):
        t = int(arg)
        if t == 0:
            return lambda T, U: (T + U) & MASK
        def f(T, U, t=t, kind=kind):
            inc = (U >> t) + ({"round": (U >> (t - 1)) & 1, "tplus": 1}.get(kind, 0))
            return (((T >> t) + inc) << t) & MASK
        return f
    if kind == "evo":
        _, cellf = load_cell(evo_source(arg).with_suffix(".c"))
        k = evo_support(arg) or evo_width(arg)
        q = 1 << k
        x, u = np.meshgrid(np.arange(q), np.arange(q), indexing="ij")
        o = cellf(x.ravel(), u.ravel())  # zero high bits: o = q*c + z by the residue property
        zt, ct = o % q, o >> k
        def f(T, U, k=k, q=q, zt=zt, ct=ct):
            idx = (T & (q - 1)) * q + (U & (q - 1))
            return ((((T >> k) + (U >> k) + ct[idx]) << k) | zt[idx]) & MASK
        return f
    if kind in CLASSIC:
        k = int(arg); py = CLASSIC[kind][1]; q = 1 << k
        def f(T, U, k=k, py=py, q=q):
            z, c = py(T & (q - 1), U & (q - 1), k)
            return ((((T >> k) + (U >> k) + c) << k) | z) & MASK
        return f
    raise ValueError(design)
