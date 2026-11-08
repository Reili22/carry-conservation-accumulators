"""Exact metrics under the operand law of the hardware traces (products of independent operands).

The MAC traces of netsim.mac_trace multiply independent uniform operands, so
consecutive residues are iid with the exact law of (a*b) mod 256 below, and
Theorem 3 applies exactly.  Adds, per design, the diffusion and compensated
MSE at n under that law ("_prod" fields), for the unsigned and signed traces.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from theory import chain_analysis, low_tables, moments

HERE = Path(__file__).parent


@lru_cache(maxsize=2)
def product_pmf(signed):
    v = np.arange(-127, 128) if signed else np.arange(256)
    a, b = np.meshgrid(v, v, indexing="ij")
    return np.bincount(((a * b) & 255).ravel(), minlength=256).astype(float)


def law_metrics(design, signed, ns=(144, 576)):
    k, z, c = low_tables(design)
    q = 1 << k
    if q == 1:
        return {"diffusion_prod": 0.0, **{f"mse_comp_{n}_prod": 0.0 for n in ns}, "drift_prod": 0.0}
    pmf = product_pmf(signed).reshape(-1, q).sum(0)
    ca = chain_analysis(z, c, pmf=pmf)
    mo = moments(z, c, list(ns), pmf=pmf)
    out = {"diffusion_prod": ca["diffusion"], "drift_prod": ca["drift"], "classes_prod": ca["classes"]}
    for n in ns:
        mean, var = mo[n]
        out[f"mse_comp_{n}_prod"] = var + (mean - round(mean)) ** 2
    return out


def main():
    for tag, signed in (("u", False), ("s", True)):
        p = HERE / f"hw_study_{tag}.json"
        rows = json.loads(p.read_text())
        for r in rows:
            r.update(law_metrics(r["design"], signed))
        p.write_text(json.dumps(rows, indent=1))
        multi = [r["design"] for r in rows if r.get("classes_prod", 1) != 1]
        pm = product_pmf(signed)
        print(tag, "updated", len(rows), "; several recurrent classes under product law:", multi,
              "; P(residue=0 mod 256) = %.3f, LSB=1 prob %.3f" % (pm[0] / pm.sum(), pm[1::2].sum() / pm.sum()))


if __name__ == "__main__":
    main()
