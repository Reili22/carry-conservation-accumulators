"""Hardware comparison of approximate low parts against exact rounding/truncation accumulators.

For each approximate design and each error metric (diffusion, or compensated
MSE at the accumulation length n), the references are exact, round_t and
trunc_t accumulators whose metric is no larger.  Reported:
  per-metric : best area / energy / delay of the design vs best of references
  logic-only : as above for area without registers
  iso-delay  : for every mapped point, the cheapest reference point no slower
  joint      : number of the design's mapped points not dominated in
               (area, delay, energy) by any reference point
Energy is the unit-delay (glitch-aware) energy per accumulation.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
REF = ("exact", "round_", "trunc_", "tplus_")


def load(tag):
    return json.loads((HERE / f"hw_study_{tag}.json").read_text())


FF_AREA = 4.522  # DFF_X1


def pts(r):
    return [(m["area_um2"], m["delay_ps"], m["e_unit_fJ"], m["logic_area_um2"]) for m in r["hw"].values()]


def twin_points(tag):
    """Pruned, formally verified state-blind netlists are also implementations of their round/trunc twin."""
    p = HERE / f"prune_check_{tag}.json"
    out = {}
    if p.exists():
        for x in json.loads(p.read_text()):
            if x["twin"] and x["twin_cec"] and x["twin_matches"]:
                out.setdefault(x["twin"], []).append(
                    (x["area"], x["delay"], x["energy"], x["area"] - x["ff"] * FF_AREA))
    return out


def analyse(rows, metric, extra=None):
    """Finite-length metrics use a 1% tie tolerance: a design may differ from its quantizer
    twin by the bounded readout term, a relative O(1/n) effect (Theorem 1)."""
    tol = 0.01 if metric.startswith("mse") else 1e-6  # 1e-6: floating-point noise of the Poisson solve
    extra = extra or {}
    refs = [r for r in rows if r["design"].startswith(REF)]
    apx = [r for r in rows if not r["design"].startswith(REF) and r["k"] > 0]
    out = []
    for r in apx:
        ok = [x for x in refs if x[metric] <= r[metric] * (1 + tol) + 1e-9]
        P, RP = pts(r), [p for x in ok for p in pts(x) + extra.get(x["design"], [])]
        best = lambda S, i: min(p[i] for p in S)  # noqa: E731
        # measurement resolution: area summation rounding 0.01 um^2, ABC timing print 0.5 ps
        A_EPS, D_EPS, E_EPS = 0.01, 0.5, 1e-6
        iso = []
        for p in P:
            cand = [c for c in RP if c[1] <= p[1] + D_EPS]
            if cand:
                iso.append((p[0] / min(c[0] for c in cand) - 1, p[2] / min(c[2] for c in cand) - 1))
        joint = sum(1 for p in P if not any(c[0] <= p[0] + A_EPS and c[1] <= p[1] + D_EPS and
                                            c[2] <= p[2] * (1 + E_EPS) for c in RP))
        out.append({"design": r["design"], "blind": r["state_blind"],
                    "area": best(P, 0) / best(RP, 0) - 1, "energy": best(P, 2) / best(RP, 2) - 1,
                    "delay": best(P, 1) / best(RP, 1) - 1, "logic": best(P, 3) / best(RP, 3) - 1,
                    "iso_area_min": min(a for a, _ in iso) if iso else None,
                    "iso_energy_min": min(e for _, e in iso) if iso else None,
                    "iso_points_without_faster_ref": len(P) - len(iso), "joint_undominated": joint})
    return out


def summary(res, label):
    for blind in (True, False):
        R = [x for x in res if x["blind"] == blind]
        f = lambda k: np.array([x[k] for x in R])  # noqa: E731
        a, e, d, lg = f("area"), f("energy"), f("delay"), f("logic")
        print(f"  {label} {'blind' if blind else 'aware'} n={len(R)}: area med {100*np.median(a):.1f}% "
              f"min {100*a.min():.1f}% | energy med {100*np.median(e):.1f}% min {100*e.min():.1f}% | "
              f"delay med {100*np.median(d):.1f}% min {100*d.min():.1f}% | logic-only med {100*np.median(lg):.1f}% "
              f"min {100*lg.min():.1f}% (neg: {[x['design'] for x in R if x['logic'] < 0]})")
        neg = [x["design"] for x in R if x["area"] < 0 or x["energy"] < 0]
        joint = [(x["design"], x["joint_undominated"]) for x in R if x["joint_undominated"]]
        iso_win = [x["design"] for x in R if x["iso_area_min"] is not None and
                   (x["iso_area_min"] < 0 and x["iso_energy_min"] < 0)]
        print(f"      better best-area or best-energy: {neg}")
        print(f"      joint-undominated points: {joint}")
        print(f"      iso-delay point cheaper in both area and energy: {iso_win}")


def main():
    report = {}
    for tag in sys.argv[1:] or ["u", "s"]:
        rows = load(tag)
        extra = twin_points(tag)
        report[tag] = {}
        for metric in ("diffusion", "mse_comp_144", "mse_comp_576", "diffusion_prod", "mse_comp_144_prod", "mse_comp_576_prod"):
            res = analyse(rows, metric, extra)
            report[tag][metric] = res
            print(f"[{tag}] metric={metric}")
            summary(res, metric)
    (HERE / "hw_analysis.json").write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
