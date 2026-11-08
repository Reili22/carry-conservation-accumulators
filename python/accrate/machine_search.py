"""Search for law-optimal small low-part machines (open question of Section 8).

For k bits (q = 2^k) and S states, a machine is a next-state table f (S x q)
and a carry table c (S x q, values 0/1).  Simulated annealing minimizes the
exact diffusion (Poisson solution, as in theory.chain_analysis) under a given
residue law; machines with several reachable recurrent classes are rejected.
Baselines: rounding that keeps log2(S) exact low bits (S = 2^s states) and the
law-optimal threshold quantizer (S = 1).
"""
import json
import sys
from pathlib import Path

import numpy as np

from product_law import product_pmf

HERE = Path(__file__).parent


def diffusion(f, c, p, q):
    S = f.shape[0]
    supp = np.nonzero(p > 0)[0]
    succ = [set(f[x, supp].tolist()) for x in range(S)]

    def reach(s):
        seen, st = {s}, [s]
        while st:
            for y in succ[st.pop()]:
                if y not in seen:
                    seen.add(y); st.append(y)
        return seen
    R0 = reach(0)
    closed = {frozenset(reach(x)) for x in R0 if all(x in reach(y) for y in reach(x))}
    if len(closed) != 1:
        return np.inf
    cls = sorted(next(iter(closed)))
    idx = {x: i for i, x in enumerate(cls)}
    n = len(cls)
    P = np.zeros((n, n)); Y = np.zeros((n, q))
    for x in cls:
        for u in supp:
            P[idx[x], idx[f[x, u]]] += p[u]
        Y[idx[x]] = q * c[x] - np.arange(q)
    ev, vec = np.linalg.eig(P.T)
    pi = np.real(vec[:, np.argmin(abs(ev - 1))]); pi /= pi.sum()
    g = Y @ p
    mu = pi @ g
    phi = np.linalg.lstsq(np.vstack([np.eye(n) - P, pi]), np.concatenate([g - mu, [0.0]]), rcond=None)[0]
    var = 0.0
    for x in cls:
        V = Y[idx[x]] + phi[[idx[f[x, u]] for u in range(q)]] * 0 + np.array(
            [phi[idx[f[x, u]]] if f[x, u] in idx else 0.0 for u in range(q)])
        m = p @ V
        var += pi[idx[x]] * float(p @ (V - m) ** 2)
    return var


def rounding_machine(k, s):
    """Exact s kept bits (state) with round-half-up of the k-s dropped bits."""
    q, S, t = 1 << k, 1 << s, k - s
    f = np.zeros((S, q), int); c = np.zeros((S, q), int)
    for x in range(S):
        for u in range(q):
            v = (x << t) + ((u >> t) << t) + ((((u >> (t - 1)) & 1) << t) if t else 0)
            f[x, u] = (v >> t) % S if s else 0
            c[x, u] = v >> k
    return f, c


def threshold_machine(k, s, p):
    """Exact s kept bits with the dropped k-s bits rounded at the law-optimal threshold (Theorem 2)."""
    q, S, t = 1 << k, 1 << s, k - s
    best = None
    for theta in range((1 << t) + 1):
        f = np.zeros((S, q), int); c = np.zeros((S, q), int)
        for x in range(S):
            for u in range(q):
                v = (x << t) + ((u >> t) << t) + ((int((u & ((1 << t) - 1)) >= theta) << t) if t else 0)
                f[x, u] = (v >> t) % S if s else 0
                c[x, u] = v >> k
        d = diffusion(f, c, p, q)
        if best is None or d < best[0]:
            best = (d, f, c, theta)
    return best


def anneal(k, S, p, iters, seed, init=None):
    rng = np.random.default_rng(seed)
    q = 1 << k
    f, c = (init if init is not None else (rng.integers(0, S, (S, q)), rng.integers(0, 2, (S, q))))
    f, c = f.copy(), c.copy()
    cur = diffusion(f, c, p, q)
    best = (cur, f.copy(), c.copy())
    T0 = max(1.0, cur if np.isfinite(cur) else 10.0)
    for it in range(iters):
        T = T0 * (1 - it / iters) ** 2 + 1e-3
        f2, c2 = f.copy(), c.copy()
        x, u = rng.integers(S), rng.integers(q)
        if rng.random() < 0.5:
            f2[x, u] = rng.integers(S)
        else:
            c2[x, u] ^= 1
        new = diffusion(f2, c2, p, q)
        if new <= cur or (np.isfinite(new) and rng.random() < np.exp(-(new - cur) / T)):
            f, c, cur = f2, c2, new
            if cur < best[0]:
                best = (cur, f.copy(), c.copy())
    return best


def main(k=4, iters=6000, restarts=6):
    q = 1 << k
    laws = {"uniform": np.full(q, 1.0 / q)}
    for signed in (False, True):
        P = product_pmf(signed).reshape(-1, q).sum(0)
        laws["signed product" if signed else "unsigned product"] = P / P.sum()
    out = {}
    for name, p in laws.items():
        out[name] = {}
        for s in (1, 2):
            S = 1 << s
            fr, cr = rounding_machine(k, s)
            d_round = diffusion(fr, cr, p, q)
            d_thr, ft, ct, theta = threshold_machine(k, s, p)
            runs = [anneal(k, S, p, iters, seed) for seed in range(restarts)]
            runs.append(anneal(k, S, p, iters, 99, init=(fr, cr)))  # start from rounding
            runs.append(anneal(k, S, p, iters, 98, init=(ft, ct)))  # start from threshold rounding
            best = min(runs, key=lambda r: r[0])
            out[name][f"S={S}"] = {"rounding": d_round, "threshold_rounding": d_thr, "theta": theta,
                                   "search_best": best[0], "gain_vs_threshold": 1 - best[0] / d_thr,
                                   "f": best[1].tolist(), "c": best[2].tolist()}
            print(f"{name:16s} S={S}: rounding {d_round:9.5f}  threshold {d_thr:9.5f}  best found "
                  f"{best[0]:9.5f}  gain vs threshold {100 * (1 - best[0] / d_thr):+.2f}%", flush=True)
    (HERE / "machine_search.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*map(int, sys.argv[1:]))
