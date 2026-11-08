"""Analysis of the bit-exact INT8 ResNet-20 accumulator study (Section 7).

Runs: cnn_raw.jsonl (no compensation), cnn_comp_uniform.jsonl (preload from the
uniform-law drift; stopped after 16 designs), cnn_cal.jsonl (preload from
Theorem 3 under each layer's residue law measured on 500 calibration images).
"""
import json
from pathlib import Path

import numpy as np

from theory import low_tables, metrics, moments

HERE = Path(__file__).parent
# state-blind or saturating designs and their quantizer twins (Theorems 1 and 2)
TWINS = {"evo_add8u_6QU": "trunc_4", "eta1_4": "trunc_4", "loawa_4": "trunc_4", "eta1_5": "trunc_5",
         "evo_add8u_0E2": "round_7", "loa_4": "round_4", "loa_5": "round_5", "ama5_4": "round_4"}
# twin up to a constant per-operation carry: identical after calibrated compensation only
TWINS_CAL = {"evo_add8u_0TP": "trunc_5"}


def load(name):
    p = HERE / name
    if not p.exists():
        return {}
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return {r["design"]: r for r in rows}


def hw_best(tag="s"):
    rows = json.loads((HERE / f"hw_study_{tag}.json").read_text())
    return {r["design"]: {k: min(m[k] for m in r["hw"].values()) for k in ("area_um2", "e_unit_fJ")} for r in rows}


def layer_prediction(design, layer, uniform=False):
    """Predicted (mean, MSE) of a layer's accumulated error from a cleared register (empirical or uniform iid law)."""
    K = layer["K"]
    if design == "exact":
        return 0.0, 0.0
    k, z, c = low_tables(design)
    q = 1 << k
    h = None if uniform else np.array(layer["hist256"], float).reshape(-1, q).sum(0)
    mean, var = moments(z, c, [K], pmf=h)[K]
    return mean, var + mean * mean


def wilson(p, n, z=1.96):
    """95% Wilson score interval for a top-1 accuracy p on n images."""
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def main():
    raw, uni, cal = load("cnn_raw.jsonl"), load("cnn_comp_uniform.jsonl"), load("cnn_cal.jsonl")
    hw = hw_best("s")  # CNN operands are signed products
    designs = list(dict.fromkeys(list(cal) + list(raw)))
    table = []
    print(f"{'design':16s} {'diffusion':>9s} {'raw':>6s} {'uniform':>8s} {'calib.':>7s} {'area':>7s} {'E':>6s}")
    for d in designs:
        if d == "int8_exact_ref":
            continue
        m = metrics(d)
        g = lambda R: R.get(d, {}).get("top1", float("nan"))  # noqa: E731
        row = {"design": d, "diffusion": m["diffusion"], "state_blind": m["state_blind"], "raw": g(raw),
               "uniform": g(uni), "calibrated": g(cal), **hw.get(d, {})}
        table.append(row)
        print(f"{d:16s} {m['diffusion']:9.2f} {row['raw']:6.3f} {row['uniform']:8.3f} {row['calibrated']:7.3f} "
              f"{row.get('area_um2', float('nan')):7.1f} {row.get('e_unit_fJ', float('nan')):6.1f}")
    print("\nTwin agreement (identical top-1 predictions on the test images):")
    twins = []
    for runs, tag, pairs in ((raw, "raw", TWINS), (cal, "calibrated", {**TWINS, **TWINS_CAL})):
        for d, t in pairs.items():
            if d in runs and t in runs:
                a, b = np.array(runs[d]["preds"]), np.array(runs[t]["preds"])
                twins.append({"mode": tag, "design": d, "twin": t, "agree": float(np.mean(a == b)),
                              "top1": runs[d]["top1"], "top1_twin": runs[t]["top1"]})
                print(f"  {tag:10s} {d:15s} vs {t:8s}: agree {twins[-1]['agree']:.3f}  "
                      f"top1 {runs[d]['top1']:.3f} vs {runs[t]['top1']:.3f}")
    print("\nPer-layer accumulated error, uncompensated runs: measured vs Theorem 3 (empirical iid law):")
    val = []
    for d, r in raw.items():
        if d == "exact":
            continue
        for name, L in r["layer_err"].items():
            pm, pmse = layer_prediction(d, L)
            um, umse = layer_prediction(d, L, uniform=True)
            if L["mse"] > 0:
                h = np.array(L["hist256"], float)
                tv = 0.5 * np.abs(h / h.sum() - 1 / 256).sum()
                val.append({"design": d, "layer": name, "K": L["K"], "mse": L["mse"], "pred": pmse,
                            "pred_uniform": umse, "mean": L["mean"], "pred_mean": pm, "tv_from_uniform": tv})
    for key, label in (("pred", "empirical law"), ("pred_uniform", "uniform law")):
        ratios = np.array([v[key] / v["mse"] for v in val])
        print(f"  {label}: {len(val)} design-layer pairs; predicted/measured MSE median {np.median(ratios):.3f}, "
              f"IQR [{np.percentile(ratios, 25):.3f}, {np.percentile(ratios, 75):.3f}], "
              f"5-95% [{np.percentile(ratios, 5):.3f}, {np.percentile(ratios, 95):.3f}]")
    tvs = sorted({(v["layer"], round(v["tv_from_uniform"], 4)) for v in val})
    print("  residue-law total variation from uniform (mod 256), per layer: min %.3f max %.3f"
          % (min(t for _, t in tvs), max(t for _, t in tvs)))
    n = next(iter(cal.values()))["n"] if cal else 1000
    print(f"  95% Wilson half-width at top-1 0.92 on n={n}: "
          f"{(wilson(0.92, n)[1] - wilson(0.92, n)[0]) / 2:.4f}")
    (HERE / "cnn_analysis.json").write_text(json.dumps({"table": table, "twins": twins, "layer_validation": val},
                                                        indent=1))


if __name__ == "__main__":
    main()
