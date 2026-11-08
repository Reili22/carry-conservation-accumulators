"""Area cost of clearing to the drift-compensation constant instead of zero (Section 6)."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import acc_designs  # noqa: E402
import asic_flow  # noqa: E402
from product_law import product_pmf  # noqa: E402
from theory import low_tables, moments  # noqa: E402

N = 144


def preload(design):
    k, z, c = low_tables(design)
    pmf = product_pmf(acc_designs.SIGNED).reshape(-1, 1 << k).sum(0)
    mean, _ = moments(z, c, [N], pmf=pmf)[N]
    return int(-round(mean)) & acc_designs.MASK


def main(designs=("round_4", "evo_add8u_6QU", "trunc_4")):
    base_rtl = acc_designs.rtl
    out = {}
    for d in designs:
        P = preload(d)
        top, text, extra = base_rtl(d)
        lo = int(re.search(r"reg \[23:(\d+)\] T;", text).group(1))
        const = f"{24 - lo}'d{P >> lo}"
        text_c = text.replace(f"module {top}(", f"module {top}_pre(").replace("clr ? 0 :", f"clr ? {const} :")
        asic_flow.rtl = lambda name, t=text_c, e=extra, tp=top: (f"{tp}_pre", t, e)
        pre = asic_flow.synth(d)
        asic_flow.rtl = base_rtl
        zero = asic_flow.synth(d)
        a0 = min(m["area_um2"] for m in zero["maps"].values())
        a1 = min(m["area_um2"] for m in pre["maps"].values())
        out[d] = {"preload": P, "area_zero": a0, "area_const": a1, "delta": a1 - a0}
        print(d, out[d])
    Path(__file__).with_name("comp_cost.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
