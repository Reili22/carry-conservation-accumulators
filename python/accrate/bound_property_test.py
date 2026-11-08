"""Property test of the state bound on random finite-state low parts.

A machine has states 0..N-1, next state f(x,u), integer carry c(x,u) in
[-2, 3] and consumes iid uniform residues u in Z_q.  The accumulated error is
sum(q c - u) (plus a bounded readout).  For every machine we compute the exact
asymptotic variance on the recurrent class reached from state 0 (Poisson
equation) and check it against
    multiplicity bound  E_pi sum_y (m_xy/q)(m_xy^2-1)/12
    state bound         ((q/S)^2-1)/12,  S = max out-degree on the class.
Also checks tightness of rounding machines (exact s-bit low part, t dropped bits).
"""
import json
from pathlib import Path

import numpy as np

rng = np.random.default_rng(2026)


def asymptotic_variance(f, c, q):
    N = f.shape[0]
    reach, st = {0}, [0]
    while st:
        for y in set(f[st.pop()].tolist()):
            if y not in reach:
                reach.add(y); st.append(y)

    def closure(s):
        seen, s2 = {s}, [s]
        while s2:
            for y in set(f[s2.pop()].tolist()):
                if y not in seen:
                    seen.add(y); s2.append(y)
        return seen
    cls = next(sorted(closure(x)) for x in sorted(reach) if all(x in closure(y) for y in closure(x)))
    idx = {x: i for i, x in enumerate(cls)}
    n = len(cls)
    P = np.zeros((n, n))
    Y = np.zeros((n, q))  # reward q c - u
    for x in cls:
        for u in range(q):
            P[idx[x], idx[f[x, u]]] += 1 / q
            Y[idx[x], u] = q * c[x, u] - u
    # stationary distribution
    w, v = np.linalg.eig(P.T)
    pi = np.real(v[:, np.argmin(abs(w - 1))]); pi /= pi.sum()
    g = Y.mean(1)
    mu = pi @ g
    # Poisson equation h - P h = g - mu, normalized pi h = 0
    M = np.vstack([np.eye(n) - P, pi])
    h = np.linalg.lstsq(M, np.concatenate([g - mu, [0]]), rcond=None)[0]
    var = 0.0
    mult = 0.0
    for x in cls:
        V = np.array([Y[idx[x], u] + h[idx[f[x, u]]] for u in range(q)])
        var += pi[idx[x]] * V.var()
        m = np.bincount([idx[f[x, u]] for u in range(q)], minlength=n)
        mult += pi[idx[x]] * sum((mm / q) * (mm * mm - 1) / 12 for mm in m if mm)
    S = max(len(set(f[x].tolist())) for x in cls)
    h = 0.0  # entropy rate H(X_i | X_{i-1}) on the same recurrent class
    for x in cls:
        m = np.bincount([idx[f[x, u]] for u in range(q)], minlength=n) / q
        m = m[m > 0]
        h += pi[idx[x]] * float(-(m * np.log2(m)).sum())
    asymptotic_variance.entropy_rate = h
    return var, mult, ((q / S) ** 2 - 1) / 12, S


def main(trials=3000):
    worst = []
    for t in range(trials):
        k = int(rng.integers(1, 5)); q = 1 << k
        N = int(rng.integers(1, 2 * q + 1))
        f = rng.integers(0, N, (N, q))
        if rng.random() < 0.3:  # structured: low-bit state with arbitrary carry
            N = q
            f = (np.arange(q)[:, None] + np.arange(q)[None, :] + rng.integers(0, q, (q, q)) * (rng.random((q, q)) < 0.3)) % q
        c = rng.integers(-2, 4, (N, q))
        var, mult, sb, S = asymptotic_variance(f, c, q)
        worst.append((var - mult, var - sb, k, N, S))
    viol_mult = sum(1 for w in worst if w[0] < -1e-7)
    viol_state = sum(1 for w in worst if w[1] < -1e-7)
    tight = []
    for k in range(1, 6):
        for s in range(0, k + 1):
            q = 1 << k; t = k - s
            # exact s-bit low part (state = high s bits of the k-bit residue), dropped t bits rounded
            N = 1 << s
            f = np.zeros((N, q), int); c = np.zeros((N, q), int)
            for x in range(N):
                for u in range(q):
                    v = (x << t) + (u >> t << t) + (((u >> (t - 1)) & 1) << t if t else 0)
                    f[x, u] = (v >> t) % N if s else 0
                    c[x, u] = v >> k
            var, mult, sb, S = asymptotic_variance(f, c, q)
            tight.append({"k": k, "s": s, "diffusion": var, "bound": ((q / (1 << s)) ** 2 - 1) / 12})
    res = {"trials": trials, "violations_multiplicity": viol_mult, "violations_state": viol_state,
           "min_slack_state": min(w[1] for w in worst),
           "rounding_tightness_max_gap": max(abs(x["diffusion"] - x["bound"]) for x in tight), "tight": tight}
    Path(__file__).with_name("bound_property_test.json").write_text(json.dumps(res, indent=1))
    print({k: v for k, v in res.items() if k != "tight"})


if __name__ == "__main__":
    main()
