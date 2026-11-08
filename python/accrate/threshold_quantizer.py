"""Theorem 5 check: for any residue law, the best state-blind design is a threshold quantizer.

A state-blind design contributes Y = q*g(R) - R per operation, so its diffusion is
Var(q g(R) - R) for iid residues with law p.  The claim: the minimum over all
integer carry functions g is attained by g(u) = 1{u >= c} for some threshold c.
Brute force over g in {-2..2}^q for random laws (q = 2, 4, 8), then the optimal
thresholds for the exact laws of the hardware traces' products.
"""
import itertools
import json
from pathlib import Path

import numpy as np

from product_law import product_pmf

rng = np.random.default_rng(11)


def var_g(g, p, q):
    v = q * np.asarray(g) - np.arange(q)
    m = p @ v
    return float(p @ (v - m) ** 2)


def best_threshold(p, q):
    vals = [(var_g((np.arange(q) >= c).astype(int), p, q), c) for c in range(q + 1)]
    return min(vals)


def brute(p, q, rng_vals=range(-2, 3)):
    return min(var_g(g, p, q) for g in itertools.product(rng_vals, repeat=q))


def main():
    worst = 0.0
    trials = 0
    for q in (2, 4, 8):
        for _ in range(60 if q < 8 else 12):
            p = rng.dirichlet(np.full(q, 0.5))
            b = brute(p, q, range(-1, 3) if q == 8 else range(-2, 3))
            t, _c = best_threshold(p, q)
            worst = max(worst, t - b)
            trials += 1
    out = {"random_law_trials": trials, "max_threshold_minus_bruteforce": worst, "product_laws": {}}
    for signed in (False, True):
        P = product_pmf(signed)
        rows = []
        for t in range(1, 9):
            q = 1 << t
            p = P.reshape(-1, q).sum(0); p = p / p.sum()
            v_opt, c = best_threshold(p, q)
            v_round = var_g((np.arange(q) >= q // 2).astype(int), p, q)
            v_trunc = var_g(np.zeros(q, int), p, q)
            rows.append({"t": t, "c_opt": int(c), "var_opt": v_opt, "var_round": v_round, "var_trunc": v_trunc})
        out["product_laws"]["signed" if signed else "unsigned"] = rows
    Path(__file__).with_name("threshold_quantizer.json").write_text(json.dumps(out, indent=1))
    print("trials", trials, "max(threshold - brute force) =", worst)
    for k, rows in out["product_laws"].items():
        print(k, [(r["t"], r["c_opt"], round(r["var_opt"], 3), round(r["var_round"], 3)) for r in rows])


if __name__ == "__main__":
    main()
