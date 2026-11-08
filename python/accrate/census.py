"""Census of lower-part adders in an accumulator (manuscript Table 1, Fig. 2, Sec. 5).

EvoApproxLib v2022 unsigned cells with residue width k <= 8 (8-bit: exhaustive;
12/16-bit: k from the SAT-certified list in carry_census.json, re-checked here by
sampling) and five classic families at k = 2..8.  Both operand orientations.
Also the ranking statistics of the published single-addition MSE against the
exact finite-length accumulated MSE at n = 256.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

from acc_designs import CLASSIC
from carry_census import tables
from evo_c2py import EVO_ROOT, cells
from theory import chain_analysis, low_tables, moments

HERE = Path(__file__).parent


def row(name, src, k, z, c, extra=None):
    q = 1 << k
    ca = chain_analysis(z, c)
    assert ca["classes"] == 1, name
    d = ca["diffusion"]
    t_eq = float(np.log2(12 * d + 1) / 2)
    r = {"design": name, "source": src, "k": k, "state_blind": bool(np.all(c == c[0:1, :])),
         "carry_range": [int(c.min()), int(c.max())], **ca, "t_eq": t_eq, "bits_bought": k - t_eq,
         "tight": bool(abs(d - ca["entropy_bound"]) < 1e-6 * max(1, d))}
    r.update(extra or {})
    return r


def main():
    certified = {r["cell"]: r for r in json.loads((HERE / "carry_census.json").read_text()) if r.get("k")}
    rows = []
    for w in (8, 12, 16):
        for name, f, meta in cells(EVO_ROOT / f"{w}_unsigned"):
            if name not in certified:
                continue
            k = certified[name]["k"]
            for side, tag in ((True, "L"), (False, "R")):
                z, c = tables(f, k, side)
                rows.append(row(name, f"EvoApproxLib {w}-bit", k, z, c,
                                {"orientation": tag, "mse1": float(meta.get("MSE", "nan"))}))
    for fam in CLASSIC:
        for k in range(2, 9):
            kk, z, c = low_tables(f"{fam}_{k}")
            rows.append(row(f"{fam}_{k}", fam, kk, z, c, {"orientation": "L"}))
    bad = [r["design"] for r in rows if r["diffusion"] < r["entropy_bound"] - 1e-6]
    (HERE / "census.json").write_text(json.dumps(rows, indent=1))
    print(f"{len(rows)} design-orientations; entropy-bound violations: {bad}")
    print(f"{'source':22s} {'n':>3s} {'blind':>5s} {'h=0':>4s} {'tight':>5s} {'max bought':>10s} "
          f"{'med eff(h)':>10s} {'max eff(h)':>10s}")
    for src in dict.fromkeys(r["source"] for r in rows):
        R = [r for r in rows if r["source"] == src and r["orientation"] == "L"]
        eff = [r["bits_bought"] / r["entropy_rate"] for r in R if r["entropy_rate"] > 1e-9 and not r["state_blind"]]
        print(f"{src:22s} {len(R):3d} {sum(r['state_blind'] for r in R):5d} "
              f"{sum(r['entropy_rate'] < 1e-9 for r in R):4d} {sum(r['tight'] for r in R):5d} "
              f"{max(r['bits_bought'] for r in R):10.2f} {np.median(eff) if eff else float('nan'):10.2f} "
              f"{max(eff) if eff else float('nan'):10.2f}")
    # orientation sensitivity
    L = {r["design"]: r for r in rows if r["orientation"] == "L" and r["source"].startswith("Evo")}
    Rr = {r["design"]: r for r in rows if r["orientation"] == "R"}
    diff = [d for d in L if abs(L[d]["diffusion"] - Rr[d]["diffusion"]) > 1e-6]
    print("cells whose diffusion depends on orientation:", diff)
    # ranking statistics (Sec. 5)
    stats = {}
    for w in (8, 12, 16):
        R = [r for r in rows if r["source"] == f"EvoApproxLib {w}-bit" and r["orientation"] == "L"]
        m1, comp, raw = [], [], []
        for r in R:
            f = dict((n, ff) for n, ff, _ in cells(EVO_ROOT / f"{w}_unsigned"))[r["design"]]
            z, c = tables(f, r["k"], True)
            mean, var = moments(z, c, [256])[256]
            m1.append(r["mse1"]); comp.append(var + (mean - round(mean)) ** 2); raw.append(var + mean ** 2)
        m1, comp, raw = map(np.array, (m1, comp, raw))
        pairs = [(i, j) for i in range(len(R)) for j in range(len(R)) if m1[i] < m1[j]]
        stats[w] = {"n": len(R), "tau_comp": kendalltau(m1, comp)[0], "tau_raw": kendalltau(m1, raw)[0],
                    "rev_comp": sum(comp[i] > comp[j] + 1e-9 for i, j in pairs),
                    "rev_raw": sum(raw[i] > raw[j] + 1e-9 for i, j in pairs), "pairs": len(pairs)}
        print(w, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in stats[w].items()})
    (HERE / "census_ranking.json").write_text(json.dumps(stats, indent=1, default=float))


if __name__ == "__main__":
    main()
