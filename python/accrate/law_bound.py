"""Law-dependent version of the grouping bound (Theorem 4, first inequality) for any residue law p.

sigma^2 >= sum_x pi(x) sum_y P_x(y) Vmin(group_xy), where group_xy is the set of
residues u with f(x,u) = y, weighted by p, and Vmin is the least weighted
variance of integers v_u = -u (mod q): by the argument of Theorem 5 it is
attained in a window [a, a+q), i.e. by the best cyclic cut of the residues.
Checked on random machines with random laws and on every design under the
exact product laws of the hardware traces.
"""
import json
from pathlib import Path

import numpy as np

from bound_property_test import asymptotic_variance
from product_law import product_pmf
from theory import chain_analysis, low_tables

rng = np.random.default_rng(13)


def vmin(us, w, q):
    """Least weighted variance of integers congruent to -u mod q (u in us, weights w)."""
    w = np.asarray(w, float) / np.sum(w)
    best = np.inf
    for a in range(q):  # window [-a, -a+q): v = -u + q*[u > a]
        v = -np.asarray(us) + q * (np.asarray(us) > a)
        m = w @ v
        best = min(best, float(w @ (v - m) ** 2))
    return best


def law_bound(f, p, q, pi):
    tot = 0.0
    for x, px in pi.items():
        for y in set(f[x].tolist()):
            us = [u for u in range(q) if f[x, u] == y and p[u] > 0]
            if us:
                wy = p[us]
                tot += px * wy.sum() * vmin(us, wy, q)
    return tot


def stationary(f, p, x0=0):
    N = f.shape[0]
    P = np.zeros((N, N))
    for x in range(N):
        for u in range(len(p)):
            P[x, f[x, u]] += p[u]
    v = np.zeros(N); v[x0] = 1; acc = np.zeros(N)
    for _ in range(3000):
        v = v @ P; acc += v
    acc /= acc.sum()
    return {x: a for x, a in enumerate(acc) if a > 1e-12}


def diffusion_law(f, c, p, q):
    """Exact diffusion of a low-part machine with states 0..N-1 under law p (single class assumed)."""
    z = f if f.shape[0] == q else None
    if z is not None:
        return chain_analysis(z, c, pmf=p)["diffusion"]
    raise ValueError


def main():
    out = {"random": 0, "random_violations": 0, "designs": {}}
    for _ in range(1500):
        k = int(rng.integers(1, 5)); q = 1 << k
        f = rng.integers(0, q, (q, q)); c = rng.integers(-1, 3, (q, q))
        p = rng.dirichlet(np.full(q, 0.7))
        ca = chain_analysis(f, c, pmf=p)
        if ca["classes"] != 1:
            continue
        b = law_bound(f, p, q, stationary(f, p))
        out["random"] += 1
        out["random_violations"] += int(ca["diffusion"] < b - 1e-6)
    viol = []
    for tag, signed in (("unsigned", False), ("signed", True)):
        P = product_pmf(signed)
        for d in [r["design"] for r in json.loads((Path(__file__).with_name("hw_study_s.json")).read_text())]:
            k, z, c = low_tables(d)
            q = 1 << k
            if q == 1:
                continue
            p = P.reshape(-1, q).sum(0); p = p / p.sum()
            ca = chain_analysis(z, c, pmf=p)
            b = law_bound(z, p, q, stationary(z, p))
            out["designs"][f"{tag}:{d}"] = {"diffusion": ca["diffusion"], "law_bound": b}
            if ca["diffusion"] < b - 1e-6:
                viol.append(f"{tag}:{d}")
    out["design_violations"] = viol
    Path(__file__).with_name("law_bound.json").write_text(json.dumps(out, indent=1))
    tight = sum(1 for v in out["designs"].values() if abs(v["diffusion"] - v["law_bound"]) < 1e-6 * max(1, v["diffusion"]))
    print(f"random machines {out['random']}, violations {out['random_violations']}; designs {len(out['designs'])}, "
          f"violations {len(viol)}, tight {tight}")


if __name__ == "__main__":
    main()
