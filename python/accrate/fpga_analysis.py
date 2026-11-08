"""KV260 (xck26) comparison of approximate low parts against exact rounding/truncation accumulators.

Accuracy matching as in hw_analysis.py and pnr_analysis.py (signed operands;
references with no larger metric, pruned state-blind twins counted as
reference implementations).  Cost: routed CLB LUTs, then CLB registers.
"""
import json
from pathlib import Path

import numpy as np

from hw_analysis import REF

HERE = Path(__file__).parent


def main():
    fp = {r["design"]: r for r in json.loads((HERE / "fpga_study.json").read_text())}
    theory = json.loads((HERE / "hw_study_s.json").read_text())
    report = {}
    for metric in ("diffusion", "diffusion_prod", "mse_comp_144_prod"):
        tol = 0.01 if metric.startswith("mse") else 1e-6
        refs = [(r["design"], r[metric]) for r in theory if r["design"].startswith(REF) and r["design"] in fp]
        refs += [(f"prune_{r['design']}", r[metric]) for r in theory if f"prune_{r['design']}" in fp]
        rows = []
        for r in theory:
            if r["design"].startswith(REF) or r["k"] == 0 or r["design"] not in fp:
                continue
            ok = [d for d, m in refs if m <= r[metric] * (1 + tol) + 1e-9]
            mine = fp[r["design"]]
            lut_ref = min(fp[d]["lut"] for d in ok)
            ff_ref = min(fp[d]["ff"] for d in ok)
            # the cheapest reference in LUTs, ties broken by registers
            best = min(ok, key=lambda d: (fp[d]["lut"], fp[d]["ff"]))
            rows.append({"design": r["design"], "blind": r["state_blind"], "lut": mine["lut"], "ff": mine["ff"],
                         "lut_ref": lut_ref, "ff_ref": ff_ref, "best_ref": best,
                         "dominated": fp[best]["lut"] <= mine["lut"] and fp[best]["ff"] <= mine["ff"],
                         "fewer_luts": mine["lut"] < lut_ref})
        report[metric] = rows
        for blind in (True, False):
            R = [x for x in rows if x["blind"] == blind]
            dl = np.array([x["lut"] - x["lut_ref"] for x in R])
            print(f"[{metric:17s}] {'blind' if blind else 'aware'} n={len(R)}: LUT excess med {np.median(dl):+.0f} "
                  f"min {dl.min():+.0f}; dominated by one reference {sum(x['dominated'] for x in R)}; "
                  f"fewer LUTs {[x['design'] for x in R if x['fewer_luts']]}")
    (HERE / "fpga_analysis.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
