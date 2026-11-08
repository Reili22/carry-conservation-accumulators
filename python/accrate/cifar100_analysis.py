"""Evaluate the pre-registered CIFAR-100 hypotheses H1-H4
(paper/integration_carry/qa/PREREGISTRATION_CIFAR100_20261030.md) on cnn100_cal_10k.jsonl.

Run with ACC_DATASET=cifar100.  H2 predicts each layer's accumulated MSE
from Theorem 3 under the layer's measured product-residue law, including the
calibrated preload that the run applied: E[(E_K + pre)^2] = Var + (mean + pre)^2.
"""
import json
import os
from pathlib import Path

import numpy as np
import torch

assert os.environ.get("ACC_DATASET") == "cifar100"
from cnn_accumulate import QNet, calibrated_preloads, residue_histograms  # noqa: E402
from resnet20 import CKPT, CLASSES, ResNet20, load_cifar  # noqa: E402
from theory import low_tables, moments  # noqa: E402
import tost  # noqa: E402
from pnr_analysis import best, load as load_pnr  # noqa: E402

HERE = Path(__file__).parent
RUN = "cnn100_cal_10k.jsonl"
TWINS = {"evo_add8u_0E2": "round_7", "evo_add8u_6QU": "trunc_4", "loa_5": "round_5"}
REF = ("exact", "round_", "trunc_")


def main():
    runs = {json.loads(l)["design"]: json.loads(l) for l in (HERE / RUN).read_text().splitlines() if l.strip()}
    out = {"n_designs": len(runs)}
    # H1
    h1 = {d: float(np.mean(np.array(runs[d]["preds"]) == np.array(runs[t]["preds"]))) for d, t in TWINS.items()}
    out["H1"] = {"agreement": h1, "pass": all(v >= 0.98 for v in h1.values())}
    # H2: rebuild the calibration exactly as cnn_accumulate.main did
    torch.set_num_threads(8)
    model = ResNet20(classes=CLASSES)
    model.load_state_dict(torch.load(CKPT, weights_only=True)["state_dict"])
    model.eval()
    xtr, _ = load_cifar(True)
    g = torch.Generator().manual_seed(1)
    calib = xtr[torch.randperm(len(xtr), generator=g)[:500]]
    with torch.no_grad():
        hist = residue_histograms(QNet(model, calib), calib)
    ratios = []
    for d, r in runs.items():
        if d == "exact":
            continue
        k, z, c = low_tables(d)
        q = 1 << k
        pre = calibrated_preloads(d, hist)
        for name, L in r["layer_err"].items():
            pmf = np.array(L["hist256"], float).reshape(-1, q).sum(0)
            mean, var = moments(z, c, [L["K"]], pmf=pmf)[L["K"]]
            pred = var + (mean + pre.get(name, 0)) ** 2
            if L["mse"] > 0:
                ratios.append(pred / L["mse"])
    ratios = np.array(ratios)
    out["H2"] = {"pairs": len(ratios), "median": float(np.median(ratios)),
                 "iqr": [float(np.percentile(ratios, 25)), float(np.percentile(ratios, 75))],
                 "pass": bool(0.9 <= np.median(ratios) <= 1.1)}
    # H3: paired TOST, +-1 pp (primary) and +-0.5 pp (secondary)
    tost.main(RUN, 0.01)
    t1 = json.loads((HERE / RUN.replace(".jsonl", "_tost.json")).read_text())
    tost.main(RUN, 0.005)
    t05 = json.loads((HERE / RUN.replace(".jsonl", "_tost.json")).read_text())
    (HERE / RUN.replace(".jsonl", "_tost_half_point.json")).write_text(json.dumps(t05, indent=1))
    tost.main(RUN, 0.01)  # leave the primary analysis in *_tost.json
    eq = sorted(d for d, v in t1.items() if v["equivalent"])
    out["H3"] = {"equivalent_1pp": eq, "equivalent_0.5pp": sorted(d for d, v in t05.items() if v["equivalent"]),
                 "diff_pp": {d: round(100 * v["diff"], 2) for d, v in t1.items()},
                 "ci90_pp": {d: [round(100 * x, 2) for x in v["ci90"]] for d, v in t1.items()}}
    # H4: cost ordering among equivalent designs
    synth = {r["design"]: min(m["area_um2"] for m in r["hw"].values())
             for r in json.loads((HERE / "hw_study_s.json").read_text())}
    _, pr = load_pnr()
    routed = {d: best(pr[("s", "relax", d)], "cell_area_um2") for d in runs}
    h4 = {}
    for name, area in (("synthesis", synth), ("post_route", routed)):
        refs = [d for d in eq if d.startswith(REF)]
        apx = [d for d in eq if not d.startswith(REF)]
        cheapest = min(refs, key=lambda d: area[d])
        h4[name] = {"cheapest_reference": cheapest, "area": area[cheapest],
                    "approx_cheaper": [d for d in apx if area[d] < area[cheapest]],
                    "areas": {d: area[d] for d in eq}}
    out["H4"] = {**h4, "pass": all(not v["approx_cheaper"] for v in h4.values())}
    out["top1"] = {d: r["top1"] for d, r in runs.items()}
    (HERE / "cifar100_analysis.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "H4"}, indent=1))
    print("H4", {k: (v["cheapest_reference"], v["approx_cheaper"]) if isinstance(v, dict) else v
                 for k, v in out["H4"].items()})


if __name__ == "__main__":
    main()
