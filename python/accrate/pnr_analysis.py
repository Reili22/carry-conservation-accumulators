"""Post-route comparison of approximate low parts against exact rounding/truncation accumulators.

Same accuracy matching as hw_analysis.py: for each approximate design and
error metric, the references are the exact, round_t, trunc_t and tplus_t
accumulators whose metric is no larger.  Every design is routed at each clock
target under the same four implementation configurations (pnr_flow.CFGS);
as in the synthesis study, a design's cost is its best routed result with
non-negative setup slack.  Area is the post-route standard-cell area; energy
is OpenSTA post-route power (VCD-annotated) times the clock period.

As in hw_analysis.twin_points, the pruned state-blind netlists (prune_<d>,
Corollary 1, simulated against the twin model) are implementations of their
quantizer twin, so they count as reference implementations with the twin's
metric.  Results are also reported with the base configuration only and
without pruned twins.
"""
import json
from pathlib import Path

import numpy as np

from hw_analysis import REF

HERE = Path(__file__).parent
METRICS = ("diffusion", "mse_comp_144", "mse_comp_576", "diffusion_prod", "mse_comp_144_prod",
           "mse_comp_576_prod")


def load():
    pnr = json.loads((HERE / "pnr_study.json").read_text())
    runs = {}
    for r in pnr:
        if r["variant"] in ("u", "s"):
            runs.setdefault((r["variant"], r["clock"], r["design"]), []).append(r)
    return pnr, runs


def best(runs, key, cfgs=None):
    """Best (min) value of key over the routed configurations that meet timing."""
    ok = [r for r in runs if r.get("status") == "ok" and r.get("wns_ns", -1) >= 0 and key in r
          and (cfgs is None or r.get("cfg", "") in cfgs)]
    return min(r[key] for r in ok) if ok else None


def analyse(theory, runs, var, clock, metric, cfgs=None, twins=True):
    tol = 0.01 if metric.startswith("mse") else 1e-6
    cost = lambda d, key: best(runs.get((var, clock, d), []), key, cfgs)  # noqa: E731
    refs = [(r["design"], r[metric]) for r in theory if r["design"].startswith(REF)]
    if twins:
        refs += [(f"prune_{r['design']}", r[metric]) for r in theory if (var, clock, f"prune_{r['design']}") in runs]
    apx = [r for r in theory if not r["design"].startswith(REF) and r["k"] > 0]
    out = []
    for r in apx:
        ok = [d for d, m in refs if m <= r[metric] * (1 + tol) + 1e-9]
        row = {"design": r["design"], "blind": r["state_blind"]}
        for key, name in (("cell_area_um2", "area"), ("energy_fJ_per_op", "energy")):
            mine = cost(r["design"], key)
            cand = [(cost(d, key), d) for d in ok if cost(d, key) is not None]
            row[f"{name}_met"] = mine is not None
            if mine is not None and cand:
                v, d = min(cand)
                row[name], row[f"{name}_ref"] = mine / v - 1, d
        out.append(row)
    return out


def summarize(res):
    s = {}
    for blind in (True, False):
        R = [x for x in res if x["blind"] == blind]
        a = np.array([x["area"] for x in R if "area" in x])
        e = np.array([x["energy"] for x in R if "energy" in x])
        s["blind" if blind else "aware"] = {
            "n": len(R), "n_met_timing": sum(x["area_met"] for x in R),
            "area_med": float(np.median(a)), "area_min": float(a.min()),
            "energy_med": float(np.median(e)), "energy_min": float(e.min()),
            "cheaper_area": [x["design"] for x in R if x.get("area", 1) < 0],
            "cheaper_energy": [x["design"] for x in R if x.get("energy", 1) < 0],
            "cheaper_both": [x["design"] for x in R if x.get("area", 1) < 0 and x.get("energy", 1) < 0]}
    return s


def spread(runs):
    """Implementation scatter: relative range of routed area over the four configurations, per design."""
    out = {}
    for (var, clock, d), rs in runs.items():
        a = [r["cell_area_um2"] for r in rs if r.get("status") == "ok" and r.get("wns_ns", -1) >= 0]
        if len(a) == 4:
            out.setdefault(f"{var}/{clock}", []).append(max(a) / min(a) - 1)
    return {k: {"median": float(np.median(v)), "p90": float(np.percentile(v, 90)), "max": float(max(v))}
            for k, v in out.items()}


def prune_gain(runs):
    """Corollary 1 after routing: best area/energy of the pruned twin relative to its state-blind design."""
    out = {}
    for (var, clock, d), rs in runs.items():
        if d.startswith("prune_") and (var, clock, d[6:]) in runs:
            for key in ("cell_area_um2", "energy_fJ_per_op"):
                p, o = best(rs, key), best(runs[(var, clock, d[6:])], key)
                if p is not None and o is not None:
                    out.setdefault(f"{var}/{clock}/{key}", []).append(p / o - 1)
    return {k: {"median": float(np.median(v)), "max": float(max(v)), "min": float(min(v)), "n": len(v)}
            for k, v in out.items()}


def main():
    pnr, runs = load()
    report = {"spread": spread(runs), "prune_gain": prune_gain(runs)}
    print("implementation scatter:", json.dumps(report["spread"]))
    print("pruned twin vs state-blind design:", json.dumps(report["prune_gain"]))
    for label, cfgs, twins in (("best", None, True), ("best_no_twins", None, False), ("base_only", {""}, False)):
        for var in ("u", "s"):
            theory = json.loads((HERE / f"hw_study_{var}.json").read_text())
            for clock in ("relax", "tight"):
                for metric in METRICS:
                    res = analyse(theory, runs, var, clock, metric, cfgs, twins)
                    summ = summarize(res)
                    report[f"{label}/{var}/{clock}/{metric}"] = {"summary": summ, "designs": res}
                    if metric not in ("diffusion", "diffusion_prod", "mse_comp_144_prod"):
                        continue
                    for cls, v in summ.items():
                        print(f"[{label:13s} {var} {clock:5s} {metric:17s}] {cls} n={v['n']} met={v['n_met_timing']} "
                              f"area med {100*v['area_med']:+.1f}% min {100*v['area_min']:+.1f}% | "
                              f"energy med {100*v['energy_med']:+.1f}% min {100*v['energy_min']:+.1f}% | "
                              f"area<0 {v['cheaper_area']} energy<0 {v['cheaper_energy']}")
    # MAC context: accumulator share of a registered 8x8 signed MAC (base configuration)
    mac = {r["design"]: r for r in pnr if r["variant"] == "m" and r.get("status") == "ok"}
    if mac:
        ex = mac["exact"]["cell_area_um2"]
        report["mac"] = {d: {"cell_area_um2": r["cell_area_um2"], "vs_exact": r["cell_area_um2"] / ex - 1,
                             "wns_ns": r.get("wns_ns")} for d, r in mac.items()}
        acc_ex = best(runs[("s", "relax", "exact")], "cell_area_um2", {""})
        report["mac_accumulator_share"] = acc_ex / ex
        print(f"MAC: exact {ex:.1f} um^2, accumulator share {acc_ex / ex:.3f}; "
              + ", ".join(f"{d} {100 * v['vs_exact']:+.1f}%" for d, v in report["mac"].items()))
    (HERE / "pnr_analysis.json").write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
