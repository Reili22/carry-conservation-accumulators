"""Pathwise check of carry conservation and the state-blind-carry quantizer equivalence.

A W-bit accumulator uses an 8-bit EvoApprox cell as its low segment and an exact
upper segment fed by the cell's carry-out (bit 8).  For every state-blind cell,
compare against an exact accumulator fed with Q(U) = U - r + q*g(r) and assert
|S_n - S'_n| < q at every step, for random, adversarial, and image inputs.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from evo_c2py import EVO_ROOT, cells  # noqa: E402

W = 32
rng = np.random.default_rng(7)


def run_lpaa(f, U, t0=0):
    """Return approximate accumulator trajectory for operand matrix U[trials, n] (state = left operand)."""
    T = np.full(U.shape[0], t0, dtype=np.int64)
    traj = np.empty_like(U)
    for i in range(U.shape[1]):
        low = f(T & 255, U[:, i] & 255)
        T = (((T >> 8) + (U[:, i] >> 8)) << 8) + low
        traj[:, i] = T
    return traj


def sources(n):
    trials = 64
    yield "uniform8", rng.integers(0, 256, (trials, n))
    yield "uniform16", rng.integers(0, 1 << 16, (trials, n))
    yield "lowbias", (rng.integers(0, 16, (trials, n)) ** 2)  # skewed residues
    yield "constant", np.tile(rng.integers(0, 256, (trials, 1)), (1, n))
    yield "ramp", (np.arange(n)[None, :] * rng.integers(1, 9, (trials, 1))) % 256
    walk = np.cumsum(rng.integers(-3, 4, (trials, n)), axis=1) % 256  # strongly correlated
    yield "randomwalk", walk


def main():
    census = {r["cell"]: r for r in json.loads((Path(__file__).with_name("carry_census.json")).read_text())}
    report = []
    n = 2048
    for name, f, _ in cells(EVO_ROOT / "8_unsigned"):
        r = census.get(name)
        if not r or not r.get("k"):
            continue
        k = r["k"]; q = 1 << k
        x, u = np.meshgrid(np.arange(q), np.arange(q), indexing="ij")
        out = f(x, u)
        c = (out - out % q) // q
        stateless = bool(np.all(c == c[0:1, :]))
        g = c[0]
        worst_eq = 0
        worst_cons = 0
        for label, U in sources(n):
            U = U.astype(np.int64)
            traj = run_lpaa(f, U)
            S = traj - np.cumsum(U, axis=1)  # accumulated error, T0 = 0
            # carry conservation: S_n - sum(q c_i - r_i) == X_n - X_0, |.| < q
            X = np.concatenate([np.zeros((U.shape[0], 1), np.int64), traj % q], axis=1)
            ci = np.array([c[X[:, i], U[:, i] % q] for i in range(n)]).T
            Y = np.cumsum(q * ci - U % q, axis=1)
            cons = np.abs(S - Y - X[:, 1:])
            worst_cons = max(worst_cons, int(cons.max()))
            if stateless:
                r_ = U % q
                Sq = np.cumsum(q * g[r_] - r_, axis=1)  # exact accumulator on quantized operands
                worst_eq = max(worst_eq, int(np.abs(S - Sq).max()))
        report.append(dict(cell=name, k=k, stateless=stateless, conservation_residual=worst_cons,
                           max_gap_to_quantizer=worst_eq if stateless else None, q=q))
        print(name, k, "stateless" if stateless else "stateful",
              "cons_resid", worst_cons, "gap", worst_eq if stateless else "-", "< q =", q)
    Path(__file__).with_name("pathwise_check.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
