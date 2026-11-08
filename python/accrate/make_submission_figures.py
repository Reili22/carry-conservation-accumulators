"""Two evidence figures for the carry-conservation manuscript.

The cost plot uses the product-residue diffusion comparison with the full
reference set (rounded, truncated, and pruned quantizer twins). ASIC points
cover both operand interpretations of the same 89 designs. The FPGA file
contains LUT/FF counts but no energy measurement, so the FPGA column shows
LUT overhead only.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from acc_designs import CLASSIC, step_fn  # noqa: E402
from make_figures import (  # noqa: E402
    BLUE_DARK, CORAL_DARK, MINT_DARK, ROSE_DARK, YELLOW_DARK,
    FIG, finish_axes, save,
)

DATA = Path(__file__).resolve().parent
QUALITY = "diffusion_prod"


def _cost_data():
    hw = json.loads((DATA / "hw_analysis.json").read_text())
    pnr = json.loads((DATA / "pnr_analysis.json").read_text())
    fpga = json.loads((DATA / "fpga_analysis.json").read_text())
    rows = {}
    for flow, clock in (("Synthesis", None), ("Route 1.5 ns", "relax"),
                        ("Route 0.55 ns", "tight")):
        rows[flow] = []
        for typ in ("u", "s"):
            group = (hw[typ][QUALITY] if clock is None else
                     pnr[f"best/{typ}/{clock}/{QUALITY}"]["designs"])
            assert len(group) == 89 and len({r["design"] for r in group}) == 89
            rows[flow].extend(group)
    rows["FPGA LUT"] = fpga[QUALITY]
    assert len(rows["FPGA LUT"]) == 89
    return rows


def cost_distribution_figure():
    rows = _cost_data()
    rng = np.random.default_rng(2026)
    colors = [BLUE_DARK, MINT_DARK, CORAL_DARK, YELLOW_DARK]
    labels = ["ASIC\nsynthesis", "Route\n1.5 ns", "Route\n0.55 ns", "FPGA\nLUTs"]
    fig, axes = plt.subplots(2, 1, figsize=(6.9, 4.7), sharex=True)
    metrics = (("area", "Area / LUT overhead (%)"),
               ("energy", "Energy overhead (%)"))
    for ax, (metric, ylabel) in zip(axes, metrics):
        ax.axhline(0, color=ROSE_DARK, lw=0.9, ls="--", zorder=1)
        for pos, (flow, color) in enumerate(zip(rows, colors), start=1):
            if flow == "FPGA LUT" and metric == "energy":
                ax.text(pos, 6, "not measured", ha="center", va="center",
                        fontsize=7, color="#5C6670")
                continue
            if flow == "FPGA LUT":
                values = np.array([100 * (r["lut"] / r["lut_ref"] - 1)
                                   for r in rows[flow] if r["lut_ref"]], float)
            else:
                values = np.array([100 * r[metric] for r in rows[flow]
                                   if isinstance(r.get(metric), (int, float))], float)
            assert np.isfinite(values).all() and len(values) >= 89
            ax.scatter(pos + rng.uniform(-0.19, 0.19, len(values)), values,
                       s=6, color=color, alpha=0.28, edgecolors="none", zorder=2)
            ax.boxplot([values], positions=[pos], widths=0.38,
                       showfliers=False, patch_artist=True,
                       boxprops={"facecolor": "white", "edgecolor": color,
                                 "linewidth": 0.9},
                       medianprops={"color": "#26323C", "linewidth": 1.2},
                       whiskerprops={"color": color, "linewidth": 0.8},
                       capprops={"color": color, "linewidth": 0.8})
            ax.text(pos, 65, f"median {np.median(values):.1f}%",
                    ha="center", va="bottom", fontsize=6.5, color=color)
            print(flow, metric, "n", len(values), "median", round(float(np.median(values)), 2),
                  "min", round(float(values.min()), 2))
        ax.set_ylabel(ylabel)
        ax.set_ylim(-12, 73)
        ax.set_xlim(0.5, 4.5)
        finish_axes(ax)
    axes[1].set_xticks(range(1, 5), labels)
    axes[1].set_xlabel("Implementation")
    fig.tight_layout(h_pad=0.6)
    save(fig, "cost_distribution")


def carry_identity_figure():
    rng = np.random.default_rng(23)
    operands = rng.integers(0, 256, size=96, dtype=np.int64)
    designs = (("loawa_4", "LOAWA: state-blind carry"),
               ("loa_4", "LOA: state-aware carry"),
               ("ama5_4", "AMA5: delayed carry"))
    fig, axes = plt.subplots(3, 1, figsize=(6.8, 4.6), sharex=True)
    for ax, (name, title) in zip(axes, designs):
        kind, _, width = name.partition("_")
        k = int(width)
        q = 1 << k
        low = CLASSIC[kind][1]
        step = step_fn(name)
        state = np.array([0], dtype=np.int64)
        actual, carry_only, reconstructed = [0], [0], [0]
        operand_total = 0
        carry_total = 0
        for u in operands:
            x = int(state[0] % q)
            r = int(u % q)
            _, c = low(np.array([x]), np.array([r]), k)
            carry_total += q * int(c[0]) - r
            state = step(state, np.array([u], dtype=np.int64))
            operand_total += int(u)
            actual.append(int(state[0]) - operand_total)
            carry_only.append(carry_total)
            reconstructed.append(carry_total + int(state[0] % q))
        assert np.array_equal(actual, reconstructed), name
        n = np.arange(len(actual))
        ax.plot(n, carry_only, ":", color=YELLOW_DARK, lw=1.0,
                label=r"$\sum(qC_i-R_i)$")
        ax.plot(n, actual, "-", color=BLUE_DARK, lw=1.3,
                label="measured $E_n$")
        ax.plot(n[::8], np.asarray(reconstructed)[::8], "o",
                ms=2.8, mfc="white", mec=ROSE_DARK, mew=0.8,
                label="carry formula + low state")
        ax.set_title(title, fontsize=8, loc="left")
        finish_axes(ax)
        print(name, "max identity residual", max(abs(a - b) for a, b in zip(actual, reconstructed)))
    axes[2].set_xlabel("Accumulations $n$")
    fig.supylabel("Accumulated error (LSB)", x=0.01, fontsize=8)
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=6.7,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(h_pad=0.35, rect=(0.06, 0.1, 1, 1))
    save(fig, "carry_identity")


if __name__ == "__main__":
    cost_distribution_figure()
    carry_identity_figure()
    print("figures in", FIG)
