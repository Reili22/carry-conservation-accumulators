"""Full accumulator study: exact theory metrics plus Nangate45 curves for every design."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from acc_designs import CLASSIC, EVO_ROOT, SIGNED  # noqa: E402
import asic_flow  # noqa: E402
from theory import metrics  # noqa: E402


def designs():
    ds = ["exact"] + [f"{f}_{t}" for f in ("round", "trunc") for t in range(1, 9)]
    ds += [f"evo_{p.stem}" for p in sorted((EVO_ROOT / "8_unsigned").rglob("*.v"))
           if not p.stem.endswith("_pdk45")]
    cert = json.loads((Path(__file__).with_name("carry_census.json")).read_text())
    ds += [f"evo_{r['cell']}" for r in cert if r["w"] in (12, 16) and r.get("k")]
    ds += [f"{fam}_{k}" for fam in CLASSIC for k in range(2, 9)]
    return list(dict.fromkeys(ds))


def main():
    ds = designs()
    theory = {d: metrics(d) for d in ds}
    hw = {r["design"]: r for r in asic_flow.main(ds, workers=10)}
    rows = []
    for d in ds:
        r = dict(theory[d])
        r["hw"] = {t: {k: v for k, v in m.items() if k != "netlist"} for t, m in hw[d]["maps"].items()}
        r["all_match"] = hw[d]["all_match"]
        r["all_cec"] = hw[d]["all_cec"]
        r["log_sha256"] = hw[d]["log_sha256"]
        rows.append(r)
    out = Path(__file__).with_name("hw_study_s.json" if SIGNED else "hw_study_u.json")
    out.write_text(json.dumps(rows, indent=1))
    bad = [r["design"] for r in rows if not (r["all_match"] and r["all_cec"])]
    print(f"{len(rows)} designs (signed={SIGNED}), simulation or equivalence failures: {bad}")


if __name__ == "__main__":
    main()
