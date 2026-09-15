"""Matched selected-box-count figures.

Reads   data/matched_box_count_pairs.csv          (20 settings)
Writes  outputs/figures/fig_matched_box_count_lost_ratio.{pdf,png}
        outputs/figures/fig_matched_box_count_metric_deltas.{pdf,png}

Figure 1 plots the lost-ratio of the density gate and of the box-count-matched
distance-only rule against the outer distance threshold, one panel per density
percentile. Figure 2 plots the per-setting difference (density minus
distance-only) for lost-ratio, mAP@0.5 and native nuScenes mAP, which is where
the comparison becomes metric-specific.

Values are read verbatim from the shipped CSV; nothing is recomputed.

Run:  python scripts/make_matched_box_count_figures.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _common import DISTANCES, FIG_OUT, PERCENTILES, fmt_d, load

DPI = 300
DENSITY_C = "#0C6E77"
DISTONLY_C = "#545E58"
PCT_C = {50: "#8C5A3C", 70: "#B08A2E", 80: "#3E6E9E", 90: "#0C6E77"}

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9.5,
    "figure.dpi": 110, "savefig.dpi": DPI,
})


def save(fig, stem):
    for ext in ("pdf", "png"):
        p = os.path.join(FIG_OUT, f"{stem}.{ext}")
        fig.savefig(p, bbox_inches="tight")
        print("wrote", os.path.join("outputs", "figures", f"{stem}.{ext}"))
    plt.close(fig)


def main():
    rows = load("matched_box_count_pairs.csv")
    by = {(float(r["D"]), int(r["percentile"])): r for r in rows}

    # ---- Figure 1: lost-ratio levels, one panel per percentile ---------
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.5), sharex=True)
    for ax, p in zip(axes, PERCENTILES):
        den = [float(by[(d, p)]["density_lost_ratio"]) for d in DISTANCES]
        dist = [float(by[(d, p)]["do_lost_ratio"]) for d in DISTANCES]
        ax.plot(DISTANCES, den, marker="o", ms=5, lw=1.5, color=DENSITY_C,
                label="distance+density")
        ax.plot(DISTANCES, dist, marker="s", ms=5, lw=1.5, color=DISTONLY_C,
                label="distance-only, box-count matched")
        ax.set_title(f"$T_{{\\rho}}$ = p{p}")
        ax.set_xlabel("$T_{\\mathrm{dist}}$ (m)")
        ax.set_xticks(DISTANCES)
        ax.set_xticklabels([fmt_d(d) for d in DISTANCES], fontsize=8.5)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Lost-ratio @ IoU 0.5")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Matched selected-box count: the lower distance-only threshold has the "
                 "lower lost-ratio in all 20 settings", fontsize=11.5)
    fig.tight_layout()
    save(fig, "fig_matched_box_count_lost_ratio")

    # ---- Figure 2: per-setting deltas, three metrics -------------------
    panels = [("delta_lost_ratio", "$\\Delta$ lost-ratio", "negative favours density"),
              ("delta_mAP50", "$\\Delta$ mAP@0.5", "positive favours density"),
              ("delta_native_mAP", "$\\Delta$ native nuScenes mAP", "positive favours density")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))
    width = 0.2
    for ax, (col, ylab, note) in zip(axes, panels):
        for k, p in enumerate(PERCENTILES):
            xs = [i + (k - 1.5) * width for i in range(len(DISTANCES))]
            ys = [float(by[(d, p)][col]) for d in DISTANCES]
            ax.bar(xs, ys, width=width, color=PCT_C[p], label=f"p{p}")
        ax.axhline(0, color="#333", lw=0.9)
        ax.set_xticks(range(len(DISTANCES)))
        ax.set_xticklabels([fmt_d(d) for d in DISTANCES])
        ax.set_xlabel("$T_{\\mathrm{dist}}$ (m)")
        ax.set_ylabel(ylab)
        ax.set_title(f"{ylab}  ({note})", fontsize=10.5)
        ax.grid(alpha=0.25, axis="y")
    axes[0].legend(frameon=False, ncol=4, fontsize=9)
    fig.suptitle("Density minus box-count-matched distance-only, per setting", fontsize=11.5)
    fig.tight_layout()
    save(fig, "fig_matched_box_count_metric_deltas")


if __name__ == "__main__":
    main()
