"""Figures for the carry-conservation manuscript (PDF + PNG into paper/integration_carry/figures)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE
FIG = HERE.parents[1] / "paper" / "integration_carry" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.size": 8,
    "font.family": "serif",
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#5C6670",
    "axes.labelcolor": "#26323C",
    "text.color": "#26323C",
    "xtick.color": "#5C6670",
    "ytick.color": "#5C6670",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

# Five-color palette; darker strokes keep pale marks legible.
BLUE, MINT, YELLOW, CORAL, ROSE = (
    "#74A9C5", "#C2E5CF", "#EDDDAB", "#F2B8AE", "#DD7389"
)
BLUE_DARK, MINT_DARK, YELLOW_DARK, CORAL_DARK, ROSE_DARK = (
    "#477F9C", "#65A889", "#A88C4D", "#C97970", "#A74F68"
)
BLIND, AWARE, REF = BLUE, ROSE, YELLOW


def finish_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E9EDF0", linewidth=0.45)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def state_bits_figure():
    """Rounding-equivalent accuracy gain versus low-part state entropy rate."""
    rows = [r for r in json.loads((DATA / "census.json").read_text()) if r["orientation"] == "L"]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    s = np.linspace(0, 8.4, 50)
    ax.plot(s, s, color=YELLOW_DARK, lw=1.15, label="bound (Thm. 5) = rounding")
    ax.fill_between(s, s, 8.4, color=YELLOW, alpha=0.36, lw=0)
    ax.text(5.45, 7.75, "unattainable", fontsize=7, color=YELLOW_DARK)
    rng = np.random.default_rng(0)
    for lib, mk in ((True, "o"), (False, "s")):
        for blind, col, edge in ((True, BLIND, BLUE_DARK), (False, AWARE, ROSE_DARK)):
            P = np.array([(r["entropy_rate"], r["bits_bought"]) for r in rows
                          if r["source"].startswith("Evo") == lib and r["state_blind"] == blind] or np.zeros((0, 2)))
            if not len(P):
                continue
            jit = rng.uniform(-0.05, 0.05, P.shape)
            ax.scatter(P[:, 0] + jit[:, 0], P[:, 1] + jit[:, 1], s=17, marker=mk,
                       facecolors="none" if blind else col, edgecolors=edge, linewidths=0.75,
                       label=f"{'EvoApproxLib' if lib else 'classic'}, {'state-blind' if blind else 'state-aware'}")
    ax.set_xlabel(r"entropy rate of the low-part state $h$ (bits/op)")
    ax.set_ylabel(r"accuracy gain, $k-t_{\mathrm{eq}}$ (bits)")
    ax.set_xlim(-0.3, 8.4); ax.set_ylim(-0.4, 8.4)
    finish_axes(ax)
    ax.legend(loc="upper left", fontsize=6.2, bbox_to_anchor=(0.0, 0.9))
    save(fig, "state_bits")
    return rows


def hw_figure(tag="u"):
    rows = json.loads((DATA / f"hw_study_{tag}.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.3), sharey=True)
    axes_keys = (("area_um2", r"area ($\mu$m$^2$)"), ("e_unit_fJ", "energy per accumulation (fJ)"),
                 ("delay_ps", "min. delay (ps)"))
    for ax, (key, lab) in zip(axes, axes_keys):
        for fam, col, mk in (("round_", YELLOW_DARK, "-"), ("trunc_", MINT_DARK, "--")):
            R = sorted([r for r in rows if r["design"].startswith(fam)] + [r for r in rows if r["design"] == "exact"],
                       key=lambda r: r["diffusion"])
            xs = [min(m[key] for m in r["hw"].values()) for r in R]
            ys = [max(r["diffusion"], 1e-2) for r in R]
            ax.plot(xs, ys, mk, color=col, lw=1.2, marker="o", ms=3,
                    markerfacecolor=YELLOW if fam == "round_" else MINT,
                    markeredgecolor=col, markeredgewidth=0.55,
                    label="rounding $t$=0..8" if fam == "round_" else "truncation")
        for r in rows:
            if r["design"].startswith(("round_", "trunc_", "exact")) or r["k"] == 0:
                continue
            x = min(m[key] for m in r["hw"].values())
            blind = r["state_blind"]
            ax.scatter(x, max(r["diffusion"], 1e-2), s=15, marker="o",
                       facecolors="none" if blind else AWARE,
                       edgecolors=BLUE_DARK if blind else ROSE_DARK, linewidths=0.65)
        ax.set_xlabel(lab)
        ax.set_yscale("log")
        finish_axes(ax)
    axes[0].set_ylabel(r"diffusion $\sigma^2_\infty$ (LSB$^2$/op)")
    axes[0].scatter([], [], s=15, facecolors="none", edgecolors=BLUE_DARK, label="state-blind")
    axes[0].scatter([], [], s=15, color=AWARE, edgecolors=ROSE_DARK, label="state-aware")
    axes[0].legend(fontsize=6.2, loc="lower left")
    fig.tight_layout(w_pad=0.6)
    save(fig, "hw_pareto" if tag == "u" else "hw_pareto_signed")


def cnn_figure():
    """Top-1 accuracy versus best signed-accumulator area, uncompensated and data-calibrated."""
    tab = json.loads((DATA / "cnn_analysis.json").read_text())["table"]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.5))
    for ax, key, title in zip(axes, ("raw", "calibrated"), ("no compensation", "data-calibrated compensation")):
        for fam, col, ls in (("round_", YELLOW_DARK, "-"), ("trunc_", MINT_DARK, "--")):
            R = sorted([r for r in tab if r["design"].startswith(fam) and r[key] == r[key]], key=lambda r: r["area_um2"])
            ax.plot([r["area_um2"] for r in R], [100 * r[key] for r in R], ls,
                    color=col, marker="o", ms=3, lw=1.2,
                    markerfacecolor=YELLOW if fam == "round_" else MINT,
                    markeredgecolor=col, markeredgewidth=0.55,
                    label="rounding" if fam == "round_" else "truncation")
        for r in tab:
            if r["design"].startswith(("round_", "trunc_", "exact")) or r[key] != r[key]:
                continue
            blind = r["state_blind"]
            ax.scatter(r["area_um2"], 100 * r[key], s=17, facecolors="none" if blind else AWARE,
                       edgecolors=BLUE_DARK if blind else ROSE_DARK, linewidths=0.7)
        ex = next(r for r in tab if r["design"] == "exact")
        ax.scatter(ex["area_um2"], 100 * ex[key], marker="*", s=48,
                   color=CORAL, edgecolors=CORAL_DARK, linewidths=0.7, label="exact")
        ax.set_title(title, fontsize=8)
        ax.set_xlabel(r"accumulator area ($\mu$m$^2$)")
        finish_axes(ax)
    axes[0].set_ylabel("top-1 accuracy (%)")
    axes[1].set_ylim(90, 93.5)
    axes[1].set_ylabel("top-1 accuracy (%)")
    axes[1].annotate("trunc_8 below scale (37.9%)", xy=(0.02, 0.02),
                     xycoords="axes fraction", fontsize=5.5)
    axes[0].scatter([], [], s=17, facecolors="none", edgecolors=BLUE_DARK, label="state-blind")
    axes[0].scatter([], [], s=17, color=AWARE, edgecolors=ROSE_DARK, label="state-aware")
    axes[0].legend(fontsize=6.2, loc="lower right")
    fig.tight_layout(w_pad=0.6)
    save(fig, "cnn_accuracy")


if __name__ == "__main__":
    rows = state_bits_figure()
    print(len(rows), "points; above bound:",
          sum(1 for r in rows if r["bits_bought"] > r["entropy_rate"] + 1e-6))
    hw_figure()
    cnn_figure()
    print("figures in", FIG)
