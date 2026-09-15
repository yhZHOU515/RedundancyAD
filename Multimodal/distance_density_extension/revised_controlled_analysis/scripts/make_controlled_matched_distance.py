"""Controlled matched-distance figure and table.

Reads   data/controlled_matched_distance.csv
Writes  outputs/figures/fig_controlled_matched_distance.{pdf,png}
        outputs/tables/table_controlled_matched_distance.{csv,tex}

This is the matched-distance-threshold comparison as evaluated under the common
inference setup: at each outer threshold the full distance-only rule is compared
with the p80 and p90 density gates, all against one shared saved full-sensor
baseline. At a matched threshold the density gate restricts the candidate set, so
it selects fewer boxes; the amount removed is therefore not held constant here.
The matched selected-box-count and matched-point-budget comparisons in this
directory are what hold the amount removed fixed.

Values are read verbatim from the shipped CSV; nothing is recomputed.

Run:  python scripts/make_controlled_matched_distance.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _common import DISTANCES, FIG_OUT, TAB_OUT, fmt_d, load, write_csv, write_tex

DPI = 300
SERIES = [("distance_only", "", "distance-only", "#B3402F", "o"),
          ("distance_density", "80", "distance+density p80", "#3E6E9E", "s"),
          ("distance_density", "90", "distance+density p90", "#0C6E77", "^")]

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 11.5, "axes.labelsize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9.5, "legend.fontsize": 9.5,
    "figure.dpi": 110, "savefig.dpi": DPI,
})


def main():
    rows = load("controlled_matched_distance.csv")
    base = [r for r in rows if r["method"] == "full_sensor_baseline"][0]
    by = {(r["method"], r["density_percentile"], float(r["T_dist_m"])): r
          for r in rows if r["method"] != "full_sensor_baseline"}

    # ---- table ---------------------------------------------------------
    header = ["T_dist_m", "rule", "selected_boxes", "selected_pct_of_eligible_pool",
              "dedup_points_removed", "dedup_points_removed_pct_of_holdout",
              "lost_ratio", "mAP50", "native_mAP"]
    out = [["baseline", "full-sensor baseline", "0", "0.00", "0", "0.00",
            "%.5f" % float(base["lost_ratio"]), "%.5f" % float(base["mAP50"]),
            "%.5f" % float(base["native_mAP"])]]
    for d in DISTANCES:
        for meth, pctl, label, _c, _m in SERIES:
            r = by[(meth, pctl, d)]
            out.append([fmt_d(d), label, r["selected_box_count"],
                        "%.2f" % float(r["selected_pct_of_eligible_pool"]),
                        r["dedup_points_removed"],
                        "%.2f" % float(r["dedup_points_removed_pct_of_holdout"]),
                        "%.5f" % float(r["lost_ratio"]), "%.5f" % float(r["mAP50"]),
                        "%.5f" % float(r["native_mAP"])])
    write_csv(os.path.join(TAB_OUT, "table_controlled_matched_distance.csv"), header, out)

    tex = ["% Auto-generated from data/controlled_matched_distance.csv by "
           "scripts/make_controlled_matched_distance.py",
           r"\begin{table}[t]", r"  \centering",
           r"  \caption{Controlled matched-distance comparison on the BEVFusion 25-scene "
           r"nuScenes holdout, all conditions evaluated against one shared full-sensor "
           r"baseline under a fixed inference seed. ``Boxes'' is the percentage of the "
           r"$74{,}464$-object eligible pool selected; ``Points'' is the percentage of "
           r"keyframe LiDAR points removed. The two are different budgets. Lost-ratio is "
           r"at IoU $\geq 0.5$ (lower is better).}",
           r"  \label{tab:controlled-matched-distance-holdout}",
           r"  \begin{tabular}{llcccc}", r"    \toprule",
           r"    $T_{\mathrm{dist}}$ & Rule & Boxes & Points & Lost-ratio & mAP@0.5 \\",
           r"    \midrule",
           "    -- & full-sensor baseline & 0.0\\%% & 0.0\\%% & %.3f & %.3f \\\\" % (
               float(base["lost_ratio"]), float(base["mAP50"]))]
    for d in DISTANCES:
        tex.append(r"    \midrule")
        for j, (meth, pctl, label, _c, _m) in enumerate(SERIES):
            r = by[(meth, pctl, d)]
            lab = (r"    \multirow{3}{*}{%s\,m}" % fmt_d(d)) if j == 0 else "    "
            tex.append("%s & %s & %.1f\\%% & %.2f\\%% & %.3f & %.3f \\\\" % (
                lab, label, float(r["selected_pct_of_eligible_pool"]),
                float(r["dedup_points_removed_pct_of_holdout"]),
                float(r["lost_ratio"]), float(r["mAP50"])))
    tex += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    write_tex(os.path.join(TAB_OUT, "table_controlled_matched_distance.tex"), tex)

    # ---- figure --------------------------------------------------------
    panels = [("selected_pct_of_eligible_pool", "Boxes selected (% of eligible pool)", None),
              ("lost_ratio", "Lost-ratio @ IoU 0.5", None),
              ("mAP50", "mAP@0.5", float(base["mAP50"])),
              ("native_mAP", "Native nuScenes mAP", float(base["native_mAP"]))]
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.6))
    for ax, (col, ylab, bl) in zip(axes.ravel(), panels):
        for meth, pctl, label, color, marker in SERIES:
            ys = [float(by[(meth, pctl, d)][col]) for d in DISTANCES]
            ax.plot(DISTANCES, ys, marker=marker, ms=5.5, lw=1.5, color=color, label=label)
        if bl is not None:
            ax.axhline(bl, color="#888", ls=":", lw=1.2, label="full-sensor baseline")
        ax.set_xlabel("$T_{\\mathrm{dist}}$ (m)")
        ax.set_ylabel(ylab)
        ax.set_xticks(DISTANCES)
        ax.set_xticklabels([fmt_d(d) for d in DISTANCES], fontsize=8.5)
        ax.grid(alpha=0.25)
    axes[0][0].legend(frameon=False, loc="upper left")
    axes[1][0].legend(frameon=False, loc="lower left")
    fig.suptitle("Controlled matched-distance comparison (amount removed is not held "
                 "constant across rules)", fontsize=11.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = os.path.join(FIG_OUT, f"fig_controlled_matched_distance.{ext}")
        fig.savefig(p, bbox_inches="tight")
        print("wrote", os.path.join("outputs", "figures",
                                    f"fig_controlled_matched_distance.{ext}"))
    plt.close(fig)


if __name__ == "__main__":
    main()
