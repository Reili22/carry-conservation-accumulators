"""Exploratory screen: single-add error metrics vs long-run accumulation rate (8-bit EvoApprox)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from evo_c2py import EVO_ROOT, cells  # noqa: E402


def residue_support(err, w):
    """Smallest k with err(a,b) a function of (a mod 2^k, b mod 2^k); None if k=w fails trivially."""
    n = 1 << w
    E = err.reshape(n, n)
    for k in range(0, w + 1):
        q = 1 << k
        T = E.reshape(n // q, q, n // q, q)
        if np.all(T == T[0:1, :, 0:1, :]):
            return k
    return w


def kernel(f, w, k, state_left=True):
    q = 1 << k
    x, u = np.meshgrid(np.arange(q), np.arange(q), indexing="ij")
    a, b = (x, u) if state_left else (u, x)
    out = f(a, b)
    e = out - a - b
    y = out % q
    return y, e  # q x q tables indexed [x, u]


def chain_stats(y, e, n_list=(1, 16, 256, 4096), x0=0):
    q = y.shape[0]
    P = np.zeros((q, q)); A = np.zeros((q, q)); B = np.zeros((q, q))
    for x in range(q):
        for u in range(q):
            P[x, y[x, u]] += 1 / q
            A[x, y[x, u]] += e[x, u] / q
            B[x, y[x, u]] += e[x, u] ** 2 / q
    v = np.zeros(q); v[x0] = 1; m = np.zeros(q); h = np.zeros(q)
    res = {}
    for i in range(1, max(n_list) + 1):
        v, m, h = v @ P, m @ P + v @ A, h @ P + 2 * m @ A + v @ B
        if i in n_list:
            mean, sq = m.sum(), h.sum()
            res[i] = (mean, sq - mean ** 2)
    n = max(n_list)
    # rate from finite differences at large n (robust to reducible/periodic chains)
    return P, res


def main():
    rows = []
    for name, f, meta in cells(EVO_ROOT / "8_unsigned"):
        w = 8
        a, b = np.meshgrid(np.arange(256), np.arange(256), indexing="ij")
        err = (f(a, b) - a - b).ravel()
        if not err.any():
            continue
        k = residue_support(err, w)
        mse1 = float(np.mean(err.astype(float) ** 2))
        var1 = float(np.var(err.astype(float)))
        row = dict(cell=name, k=k, mse1=mse1, var1=var1, mean1=float(err.mean()),
                   meta_mse=float(meta.get("MSE", "nan")), pwr=float(meta.get("PDK45_PWR", "nan")),
                   area=float(meta.get("PDK45_AREA", "nan")))
        if k <= 7:
            for side in (True, False):
                y, e = kernel(f, w, k, side)
                _, res = chain_stats(y, e)
                v4096 = res[4096][1]; v256 = res[256][1]
                tag = "L" if side else "R"
                row[f"rate_{tag}"] = (v4096 - v256) / (4096 - 256)
                row[f"drift_{tag}"] = (res[4096][0] - res[256][0]) / (4096 - 256)
        rows.append(row)
    rows.sort(key=lambda r: r["var1"])
    print(f"{"cell":12s} k  mean1    var1     rateL    rateR   rL/var1  driftL")
    for r in rows:
        if "rate_L" in r:
            print(f"{r['cell']:12s} {r['k']} {r['mean1']:7.3f} {r['var1']:8.3f} {r['rate_L']:8.3f} "
                  f"{r['rate_R']:8.3f} {r['rate_L']/r['var1']:6.2f} {r['drift_L']:7.3f}")
        else:
            print(f"{r['cell']:12s} {r['k']} {r['mean1']:7.3f} {r['var1']:8.3f}   (no small quotient)")
    Path(__file__).with_suffix(".json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
