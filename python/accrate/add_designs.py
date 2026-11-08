"""Synthesize extra designs and append them to hw_study_{u,s}.json (signedness from ACC_SIGNED)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from acc_designs import SIGNED  # noqa: E402
import asic_flow  # noqa: E402
from product_law import law_metrics  # noqa: E402
from theory import metrics  # noqa: E402


def main(designs):
    path = Path(__file__).with_name("hw_study_s.json" if SIGNED else "hw_study_u.json")
    rows = [r for r in json.loads(path.read_text()) if r["design"] not in designs]
    for hw in asic_flow.main(designs, workers=8):
        d = hw["design"]
        r = dict(metrics(d))
        r.update(law_metrics(d, SIGNED))
        r["hw"] = {t: {k: v for k, v in m.items() if k != "netlist"} for t, m in hw["maps"].items()}
        r["all_match"], r["all_cec"], r["log_sha256"] = hw["all_match"], hw["all_cec"], hw["log_sha256"]
        rows.append(r)
        print(d, "match", r["all_match"], "cec", r["all_cec"])
    path.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
