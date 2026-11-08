"""Census of EvoApproxLib unsigned adders: residue support k, carry class, accumulation rates.

Exploratory: support for w>8 is screened by random high parts (SAT proofs exist
separately for the certified set). Carry into the exact upper part is
c(x,u) = (out - out mod q - (a - x) - (b - u)) / q.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from evo_c2py import EVO_ROOT, cells  # noqa: E402

rng = np.random.default_rng(1)


def support_k(f, w, kmax=8, highs=6):
    """Smallest k whose residue pairs determine the error on sampled high parts."""
    for k in range(0, kmax + 1):
        q = 1 << k
        x, u = np.meshgrid(np.arange(q), np.arange(q), indexing="ij")
        x = x.ravel(); u = u.ravel()
        ref = None
        ok = True
        for _ in range(max(highs, 8192 // (q * q))):
            ha = rng.integers(0, 1 << (w - k), x.size) * q
            hb = rng.integers(0, 1 << (w - k), x.size) * q
            if ref is None:
                ha[:] = 0; hb[:] = 0
            a, b = x + ha, u + hb
            e = f(a, b) - a - b
            if ref is None:
                ref = e
            elif np.any(e != ref):
                ok = False
                break
        if ok:
            return k
    return None


def tables(f, k, state_left):
    q = 1 << k
    x, u = np.meshgrid(np.arange(q), np.arange(q), indexing="ij")
    a, b = (x, u) if state_left else (u, x)
    out = f(a, b)
    z = out % q
    c = (out - z) // q  # high operand parts are zero here
    return z, c


def rates(z, c, n_hi=4096, n_lo=1024):
    """Exact drift/diffusion of S_n = q C_n - sum R + X_n - X_0 via moment recursion from X_0=0."""
    q = z.shape[0]
    x_idx, u_idx = np.meshgrid(np.arange(q), np.arange(q), indexing="ij")
    e = q * c + z - x_idx - u_idx
    P = np.zeros((q, q)); A = np.zeros((q, q)); B = np.zeros((q, q))
    np.add.at(P, (x_idx, z), 1.0 / q)
    np.add.at(A, (x_idx, z), e / q)
    np.add.at(B, (x_idx, z), e.astype(float) ** 2 / q)
    v = np.zeros(q); v[0] = 1; m = np.zeros(q); h = np.zeros(q)
    out = {}
    for i in range(1, n_hi + 1):
        v, m, h = v @ P, m @ P + v @ A, h @ P + 2 * m @ A + v @ B
        if i in (n_lo, n_hi):
            out[i] = (m.sum(), h.sum() - m.sum() ** 2)
    drift = (out[n_hi][0] - out[n_lo][0]) / (n_hi - n_lo)
    diff = (out[n_hi][1] - out[n_lo][1]) / (n_hi - n_lo)
    return drift, diff


def multiplicity_bound(z):
    """Lower bound on diffusion: E_pi sum_y (m_xy/q)(m_xy^2-1)/12 with stationary pi from X_0=0 Cesaro."""
    q = z.shape[0]
    P = np.zeros((q, q))
    np.add.at(P, (np.repeat(np.arange(q), q), z.ravel()), 1.0 / q)
    v = np.zeros(q); v[0] = 1; acc = np.zeros(q)
    for _ in range(2048):
        v = v @ P; acc += v
    pi = acc / acc.sum()
    per_x = np.array([sum((m / q) * (m * m - 1) / 12 for m in np.bincount(z[x], minlength=q) if m)
                      for x in range(q)])
    return float(pi @ per_x)


def main():
    rows = []
    for width in (8, 12, 16):
        for name, f, meta in cells(EVO_ROOT / f"{width}_unsigned"):
            k = support_k(f, width)
            row = dict(cell=name, w=width, k=k, area=float(meta.get("PDK45_AREA", "nan")),
                       pwr=float(meta.get("PDK45_PWR", "nan")), mse1=float(meta.get("MSE", "nan")),
                       mae1=float(meta.get("MAE", "nan")))
            if k is not None and k > 0:
                for side, tag in ((True, "L"), (False, "R")):
                    z, c = tables(f, k, side)
                    row[f"carry_{tag}"] = ("cut" if not c.any() else
                                           "max1" if c.max() <= 1 and c.min() >= 0 else "other")
                    row[f"pcarry_{tag}"] = float(c.mean())
                    row[f"drift_{tag}"], row[f"diff_{tag}"] = rates(z, c)
                    row[f"stateless_{tag}"] = bool(np.all(c == c[0:1, :]))
                    row[f"bound_{tag}"] = multiplicity_bound(z)
            rows.append(row)
    Path(__file__).with_name("carry_census.json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        if r["k"] in (None, 0):
            print(f"{r['cell']:13s} k={r['k']}")
            continue
        q = 1 << r["k"]
        t = (q * q - 1) / 12
        print(f"{r['cell']:13s} k={r['k']} carryL={r['carry_L']:5s} pc={r['pcarry_L']:.3f} "
              f"driftL={r['drift_L']:8.3f} diffL={r['diff_L']:9.3f} diffR={r['diff_R']:9.3f} "
              f"bound={r['bound_L']:9.3f} stateless={r['stateless_L']} mse1={r['mse1']}")


if __name__ == "__main__":
    main()
