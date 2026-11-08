"""Write the supplementary post-route / FPGA / MAC tables from pnr_analysis.json and fpga_analysis.json."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OUT = HERE.parents[1] / "paper" / "integration_carry" / "pnr_tables.tex"


def pct(x):
    return f"${100 * x:+.1f}$".replace("+", "{+}") if x < 0 else f"{100 * x:.1f}"


def main():
    a = json.loads((HERE / "pnr_analysis.json").read_text())
    rows = []
    for label, name in (("best", "best of 4, with twins"), ("best_no_twins", "best of 4, no twins"),
                        ("base_only", "default only, no twins")):
        for metric, mname in (("diffusion", "uniform"), ("diffusion_prod", "product")):
            for var in ("u", "s"):
                for clock in ("relax", "tight"):
                    s = a[f"{label}/{var}/{clock}/{metric}"]["summary"]
                    cells = []
                    for cls in ("blind", "aware"):
                        v = s[cls]
                        cells += [f"{pct(v['area_med'])} ({pct(v['area_min'])})",
                                  f"{pct(v['energy_med'])} ({pct(v['energy_min'])})"]
                    rows.append(f"{name} & {mname} & {var} & {'1.5' if clock == 'relax' else '0.55'} & "
                                + " & ".join(cells) + r"\\")
    t1 = "\n".join([
        r"\begin{table}[t]", r"\centering",
        r"\caption{Post-route penalties (\%) relative to the cheapest equally accurate reference: medians "
        r"with minima in parentheses, by reference set, diffusion law, operand type (u/s), and clock (ns).}",
        r"\label{tab:pnrfull}", r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{llllrrrr}", r"\toprule",
        r"Comparison & Law & Op. & Clock & Blind area & Blind energy & Aware area & Aware energy\\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    sp = a["spread"]
    pg = a["prune_gain"]
    mac = a["mac"]
    order = ["exact", "round_5", "round_6", "round_7", "round_8", "trunc_4", "trunc_6", "trunc_7",
             "evo_add8u_0E2", "evo_add8u_6QU", "evo_add8u_2LL", "loa_5"]
    mrows = [f"\\texttt{{{d.replace('evo_', '').replace('_', chr(92) + '_')}}} & {mac[d]['cell_area_um2']:.1f} & "
             f"{pct(mac[d]['vs_exact'])}\\\\" for d in order]
    t2 = "\n".join([
        r"\begin{table}[t]", r"\centering",
        r"\caption{Routed registered $8\times8$ signed MAC (default configuration, 2~ns clock): "
        r"standard-cell area and change relative to the exact MAC (\%).}",
        r"\label{tab:mac}", r"\footnotesize", r"\begin{tabular}{lrr}", r"\toprule",
        r"Accumulator & Area ($\mu$m$^2$) & vs.\ exact\\", r"\midrule", *mrows, r"\bottomrule",
        r"\end{tabular}", r"\end{table}"])
    stats = {"spread": sp, "prune_gain": pg, "mac_share": a["mac_accumulator_share"]}
    OUT.write_text(t1 + "\n\n" + t2 + "\n")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
