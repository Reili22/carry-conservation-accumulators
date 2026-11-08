"""Dominance of approximate low parts by exact rounding/truncation accumulators.

A point (area, delay, energy) of an approximate design is dominated when some
mapped point of a reference design (exact, round_t, trunc_t) has error metric
no larger and area, delay and energy no larger.  The design is dominated when
all its mapped points are.  Error metric: long-run diffusion, or the
bias-compensated finite-length MSE at a given n.
"""
import json
import sys
from pathlib import Path

ROWS = json.loads(Path(__file__).with_name("hw_study.json").read_text())
REF = ("exact", "round_", "trunc_")
AXES = ("area_um2", "delay_ps", "e_dyn_fJ")


def points(r):
    return [tuple(m[a] for a in AXES) for m in r["hw"].values()]


def analyse(metric, tol=1e-9):
    refs = [r for r in ROWS if r["design"].startswith(REF)]
    out = []
    for r in ROWS:
        if r["design"].startswith(REF):
            continue
        err = r[metric]
        cands = [p for ref in refs if ref[metric] <= err + tol for p in points(ref)]
        surv = [p for p in points(r) if not any(all(c[i] <= p[i] for i in range(3)) for c in cands)]
        out.append((r["design"], err, r["state_blind"], len(surv), surv))
    return out


def main(metric="diffusion"):
    res = analyse(metric)
    dom = [x for x in res if x[3] == 0]
    print(f"metric={metric}: {len(dom)}/{len(res)} approximate designs fully dominated")
    for d, err, blind, ns, surv in sorted(res, key=lambda x: x[0]):
        if ns:
            best = min(surv, key=lambda p: p[0])
            print(f"  SURVIVES {d:16s} err={err:10.3f} blind={blind} points={ns} e.g. A={best[0]:.1f} D={best[1]:.0f} E={best[2]:.0f}")
    return res


if __name__ == "__main__":
    for m in sys.argv[1:] or ["diffusion", "mse_comp_256", "mse_comp_64", "mse_comp_16"]:
        main(m)
