"""Property test of Theorem 4 on random finite-state low parts (all three bound forms).

For uniform iid residues and the recurrent class reached from state 0:
    sigma^2 >= multiplicity bound >= (q^2 2^{-2h} - 1)/12 >= ((q/S)^2 - 1)/12,
with h the entropy rate on that class and S the largest out-degree there.
Every inequality of the chain is checked on every random machine; the published
and classic designs are checked by census.py (census.json, "entropy-bound violations").
"""
import json
from pathlib import Path

import numpy as np

from bound_property_test import asymptotic_variance

rng = np.random.default_rng(7)
TOL = 1e-7


def main(trials=3000):
    viol = {"var>=mult": 0, "mult>=entropy": 0, "entropy>=outdeg": 0}
    gaps = []
    for _ in range(trials):
        k = int(rng.integers(1, 5)); q = 1 << k
        N = int(rng.integers(1, 2 * q + 1))
        f = rng.integers(0, N, (N, q)); c = rng.integers(-2, 4, (N, q))
        var, mult, sb, S = asymptotic_variance(f, c, q)
        h = asymptotic_variance.entropy_rate
        eb = (q * q * 2 ** (-2 * h) - 1) / 12
        viol["var>=mult"] += int(var < mult - TOL)
        viol["mult>=entropy"] += int(mult < eb - TOL)
        viol["entropy>=outdeg"] += int(eb < sb - TOL)
        gaps.append(var - eb)
    res = {"trials": trials, "k_range": [1, 4], "violations": viol, "min_gap_var_minus_entropy_bound": min(gaps)}
    Path(__file__).with_name("entropy_bound_test.json").write_text(json.dumps(res, indent=1, default=float))
    print(res)


if __name__ == "__main__":
    main()
