"""FIR hardware comparison: area and energy on real FIR products (Section 7.5).

Energy: every mapped signed netlist is re-simulated (functional check included)
on serial FIR accumulations of products of held-out Kodak windows (images
13-24) and the K-tap filter, cleared every K operations.
Comparison: for each approximate design, the cheapest reference (exact,
round_t, trunc_t) whose measured SNR is at least the design's (strict rule),
or whose 95% bootstrap SNR interval overlaps or exceeds the design's
(tolerant rule: statistically not worse).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ["ACC_SIGNED"] = "1"
sys.path.insert(0, str(Path(__file__).parent))
from acc_designs import operand24, step_fn  # noqa: E402
from asic_flow import LIB, OUT  # noqa: E402
from fir_study import load, taps, windows  # noqa: E402
from liberty import read  # noqa: E402
import netsim  # noqa: E402

HERE = Path(__file__).parent


def fir_trace(K, n_acc=40, seed=9):
    rng = np.random.default_rng(seed)
    X = windows(load(range(13, 25)), K, n_acc, rng)
    P = (X * taps(K)[None, :]).ravel()
    clr = (np.arange(P.size) % K) == 0
    return (P & 0xFFFF).astype(np.int64), clr


def main():
    lib = read(str(LIB))
    hw = {r["design"]: r for r in json.loads((HERE / "hw_study_s.json").read_text())}
    report = {}
    for K in (64, 256):
        U16, clr = fir_trace(K)
        fir = {r["design"]: r for r in json.loads((HERE / f"fir_study_K{K}.json").read_text())}
        energy, bad = {}, []
        for d in fir:
            T = netsim.reference(step_fn(d), operand24(U16), clr)
            es = []
            for tag in hw[d]["hw"]:
                ok, en = netsim.run(netsim.parse_blif(OUT / f"acc_{d}_map_{tag}.blif", lib), lib, U16, clr, T)
                es.append(en["e_unit_fJ"])
                bad += [] if ok else [(d, tag)]
            energy[d] = min(es)
        area = {d: min(m["area_um2"] for m in hw[d]["hw"].values()) for d in fir}
        refs = [d for d in fir if d.startswith(("exact", "round_", "trunc_"))]
        rows = []
        for d, r in fir.items():
            if d in refs:
                continue
            strict = [x for x in refs if fir[x]["snr_db"] >= r["snr_db"]]
            tol = [x for x in refs if fir[x]["snr_ci95"][1] >= r["snr_ci95"][0]]
            row = {"design": d, "snr": r["snr_db"]}
            for name, S in (("strict", strict), ("tolerant", tol)):
                ra = min(S, key=lambda x: area[x]); re_ = min(S, key=lambda x: energy[x])
                row[name] = {"ref_area": ra, "area_pen": area[d] / area[ra] - 1,
                             "ref_energy": re_, "energy_pen": energy[d] / energy[re_] - 1}
            rows.append(row)
        report[K] = {"rows": rows, "mismatches": bad, "trace_ops": int(U16.size)}
        for name in ("strict", "tolerant"):
            a = [x[name]["area_pen"] for x in rows]; e = [x[name]["energy_pen"] for x in rows]
            print(f"K={K} {name:8s}: area penalty {100*min(a):+.1f}..{100*max(a):+.1f}% (median {100*np.median(a):.1f}%), "
                  f"energy {100*min(e):+.1f}..{100*max(e):+.1f}% (median {100*np.median(e):.1f}%), "
                  f"min-area design {min(rows, key=lambda x: x[name]['area_pen'])['design']}")
        print(f"K={K} trace {U16.size} ops, netlist mismatches: {bad}")
    (HERE / "fir_hw.json").write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
