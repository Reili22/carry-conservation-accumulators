"""Paired equivalence test (TOST) of top-1 accuracy against the exact accumulator.

For each design, d_i = 1{design correct} - 1{exact correct} on the same test
image; the 90% confidence interval of mean(d) inside (-margin, +margin) means
equivalence at level 0.05.  Also reports McNemar p-values.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, norm

from resnet20 import load_cifar

HERE = Path(__file__).parent


def main(fname="cnn_cal_10k.jsonl", margin=0.01):
    runs = {json.loads(l)["design"]: json.loads(l) for l in (HERE / fname).read_text().splitlines() if l.strip()}
    n = next(iter(runs.values()))["n"]
    _, y = load_cifar(False)
    y = y[:n].numpy()
    ex = np.array(runs["exact"]["preds"]) == y
    z = norm.ppf(0.95)
    out = {}
    for d, r in runs.items():
        c = np.array(r["preds"]) == y
        diff = c.astype(float) - ex.astype(float)
        m, se = diff.mean(), diff.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        lo, hi = m - z * se, m + z * se
        b, cc = int(np.sum(ex & ~c)), int(np.sum(~ex & c))
        out[d] = {"n": n, "top1": float(c.mean()), "diff": float(m), "ci90": [float(lo), float(hi)],
                  "equivalent": bool(lo > -margin and hi < margin),
                  "mcnemar_p": float(binomtest(b, b + cc).pvalue) if b + cc else 1.0,
                  "same_prediction": float(np.mean(np.array(r["preds"]) == np.array(runs["exact"]["preds"])))}
        print(f"{d:16s} top1 {c.mean():.4f} diff {100*m:+.2f} pp 90%CI [{100*lo:+.2f},{100*hi:+.2f}] "
              f"equiv(+-{100*margin:.0f}pp) {out[d]['equivalent']} McNemar p {out[d]['mcnemar_p']:.3g}")
    (HERE / fname.replace(".jsonl", "_tost.json")).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:2])
