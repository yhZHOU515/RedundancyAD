"""Holdout matched-threshold figure + table (paper-ready, self-contained).

Regenerates the manuscript holdout assets from the SHIPPED 5x5 grid only — no
raw data, checkpoints, or inference required:

  reads   data/results/holdout_grid_5x5.csv   (25 density-gate cells + 1 baseline row)
  writes  outputs/figures/fig_holdout_matched_threshold.pdf
          outputs/figures/fig_holdout_matched_threshold.png
          outputs/tables/table_holdout_matched_threshold.csv
          outputs/tables/table_holdout_matched_threshold.tex

For each strategy (distance-only / density gate p80 / p90) it plots metrics vs
the distance threshold T_dist and emits the representative matched-T_dist table
(tab:matched-tdist-holdout). Numbers are taken verbatim from the grid; nothing
is recomputed or fabricated.

Run from the package root:  python scripts/make_holdout_figure_and_table.py
"""
import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Resolve paths relative to the package root (parent of scripts/).
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GRID = os.path.join(ROOT, "data", "results", "holdout_grid_5x5.csv")
FIG_OUT = os.path.join(ROOT, "outputs", "figures")
TAB_OUT = os.path.join(ROOT, "outputs", "tables")

DPI = 300
plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "figure.dpi": DPI, "savefig.dpi": DPI, "savefig.bbox": "tight",
})

# strategies: (T_density_pct, label, color, marker)
STRATS = [
    (0,  "distance-only", "#d62728", "o"),
    (80, "density gate p80", "#1f77b4", "s"),
    (90, "density gate p90", "#2ca02c", "^"),
]
# The five distance thresholds spanned by the 5x5 grid.
T_AXIS = [10.0, 15.0, 20.0, 22.5, 30.0]


def load():
    """Return (by, base): grid cells keyed by (T_dist, density_pct), plus the
    full-sensor baseline metrics read from the tag=='baseline' row."""
    by = {}
    base = None
    for r in csv.DictReader(open(GRID)):
        if r["tag"] == "baseline":
            base = dict(mAP=float(r["mAP50"]), nat=float(r["native_mAP"]),
                        nds=float(r["native_NDS"]), ped=float(r["AP50_Ped"]))
            continue
        by[(round(float(r["T_dist"]), 1), int(r["T_density_pct"]))] = r
    if base is None:
        raise SystemExit("No baseline row in holdout_grid_5x5.csv (tag=baseline).")
    return by, base


def series(by, pct, key, scale=1.0):
    xs, ys = [], []
    for T in T_AXIS:
        r = by.get((round(T, 1), pct))
        if r:
            xs.append(T); ys.append(float(r[key]) * scale)
    return xs, ys


def make_figure(by, base):
    panels = [
        ("removal_pct_of_pool", 100.0, "objects removed (% of eligible pool)", None, "lower = less LiDAR deleted"),
        ("lost50_overall", 1.0, "lost-ratio @ IoU 0.5", 0.0, "lower = better"),
        ("mAP50", 1.0, "mAP @ IoU 0.5", base["mAP"], "higher = better"),
        ("native_mAP", 1.0, "native nuScenes mAP", base["nat"], "higher = better"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (key, scale, ylab, baseline, note) in zip(axes.flat, panels):
        for pct, lbl, col, mk in STRATS:
            xs, ys = series(by, pct, key, scale)
            if xs:
                ax.plot(xs, ys, marker=mk, color=col, lw=2.2, ms=8, label=lbl)
        if baseline is not None:
            ax.axhline(baseline, color="grey", ls=":", lw=2,
                       label=f"full-sensor baseline = {baseline:.3f}")
        ax.set_xlabel("distance threshold  $T_{dist}$  (m)")
        ax.set_ylabel(ylab)
        ax.set_title(f"{ylab}  ({note})", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(T_AXIS)
    axes.flat[0].legend(loc="upper left", framealpha=0.9)
    fig.suptitle("BEVFusion held-out 25-scene / 995-sample val: density gate vs. distance-only "
                 "at matched distance thresholds",
                 fontsize=15, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("pdf", "png"):
        out = os.path.join(FIG_OUT, f"fig_holdout_matched_threshold.{ext}")
        fig.savefig(out); print("wrote", out)
    plt.close(fig)


def make_csv(by, base):
    cols = ["T_dist_m", "strategy", "T_density_pct", "n_removed", "removal_pct",
            "lost50_overall", "mAP50", "native_mAP", "native_NDS", "AP50_Ped",
            "lost50_Ped"]
    trows = [["baseline", "no-removal", "-", 0, "0.00%",
              "0.0000", f"{base['mAP']:.4f}", f"{base['nat']:.4f}", f"{base['nds']:.4f}",
              f"{base['ped']:.4f}", "0.0000"]]
    for T in T_AXIS:
        for pct, lbl, _, _ in STRATS:
            r = by.get((round(T, 1), pct))
            if not r:
                continue
            trows.append([f"{T:.1f}", lbl, pct, int(r["n_removed"]),
                          f"{float(r['removal_pct_of_pool'])*100:.2f}%",
                          f"{float(r['lost50_overall']):.4f}", f"{float(r['mAP50']):.4f}",
                          f"{float(r['native_mAP']):.4f}", f"{float(r['native_NDS']):.4f}",
                          f"{float(r['AP50_Ped']):.4f}", f"{float(r['lost50_Ped']):.4f}"])
    out = os.path.join(TAB_OUT, "table_holdout_matched_threshold.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols); w.writerows(trows)
    print("wrote", out)


def make_tex(by, base):
    """Representative matched-T_dist comparison (tab:matched-tdist-holdout):
    baseline + distance-only vs p90 at T_dist in {10, 20, 30} m."""
    def row(T, pct):
        r = by.get((round(T, 1), pct))
        return (f"{float(r['removal_pct_of_pool'])*100:.1f}\\%",
                f"{float(r['lost50_overall']):.3f}",
                f"{float(r['mAP50']):.3f}",
                f"{float(r['native_mAP']):.3f}")
    lines = [
        "% Auto-generated from data/results/holdout_grid_5x5.csv by "
        "scripts/make_holdout_figure_and_table.py",
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Representative matched-$T_{\\mathrm{dist}}$ comparison on the BEVFusion "
        "25-scene nuScenes holdout. ``Removed'' is the percentage of the $74{,}464$-object "
        "eligible pool affected by pruning; lost-ratio is at IoU $\\geq 0.5$ (lower is better).}",
        "  \\label{tab:matched-tdist-holdout}",
        "  \\begin{tabular}{llcccc}",
        "    \\toprule",
        "    $T_{\\mathrm{dist}}$ & Rule & Removed & Lost-ratio & mAP@0.5 & Native mAP \\\\",
        "    \\midrule",
        f"    -- & Full-sensor baseline & 0.0\\% & 0.000 & {base['mAP']:.3f} & {base['nat']:.3f} \\\\",
        "    \\midrule",
    ]
    for T in (10.0, 20.0, 30.0):
        do = row(T, 0); p9 = row(T, 90)
        lines.append(f"    \\multirow{{2}}{{*}}{{{T:.0f}\\,m}} & distance-only "
                     f"& {do[0]} & {do[1]} & {do[2]} & {do[3]} \\\\")
        lines.append(f"     & distance+density p90 & {p9[0]} & {p9[1]} & {p9[2]} & {p9[3]} \\\\")
        lines.append("    \\midrule" if T != 30.0 else "    \\bottomrule")
    lines += ["  \\end{tabular}", "\\end{table}", ""]
    out = os.path.join(TAB_OUT, "table_holdout_matched_threshold.tex")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print("wrote", out)


def main():
    os.makedirs(FIG_OUT, exist_ok=True)
    os.makedirs(TAB_OUT, exist_ok=True)
    by, base = load()
    make_figure(by, base)
    make_csv(by, base)
    make_tex(by, base)


if __name__ == "__main__":
    main()
