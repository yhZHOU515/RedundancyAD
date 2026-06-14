"""Diagnostic distance--density figure + table (paper-ready, self-contained).

Regenerates the manuscript subsection "Diagnostic Analysis for Distance--Density
Pruning" (nuScenes Part 1) from the SHIPPED aggregate stats only — no raw
per-object feature table required:

  reads   data/results/diagnostic_aggregate_stats.csv
  writes  outputs/figures/fig_diagnostic_cl_redundancy.pdf
          outputs/figures/fig_diagnostic_cl_redundancy.png
          outputs/tables/table_cl_redundancy_main.csv
          outputs/tables/table_cl_redundancy_main.tex

Diagnostic only — no training/evaluation, no pipeline changes. "Supported" =
in-box LiDAR points >= 10; near cutoff = 30 m. The p80/p90 gates are PERCENTILE
thresholds on rho = n / A_2D (in-box LiDAR points / projected 2D box area), not
raw point-count thresholds. All values are read verbatim from the aggregate CSV;
nothing is recomputed from raw data or fabricated.

Run from the package root:  python scripts/make_diagnostic_figure_and_table.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# Resolve paths relative to the package root (parent of scripts/).
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
AGG = os.path.join(ROOT, "data", "results", "diagnostic_aggregate_stats.csv")
FIG_OUT = os.path.join(ROOT, "outputs", "figures")
TAB_OUT = os.path.join(ROOT, "outputs", "tables")

# Representative categories for the compact panel (span the range; include the
# manuscript-cited trailer ~427 and bicycle ~16).
CAT_SHOW = ["trailer", "truck", "car", "barrier", "motorcycle",
            "ped.adult", "bicycle", "trafficcone"]
CAM_ORDER = ["BACK_LEFT", "BACK_RIGHT", "FRONT_LEFT", "FRONT_RIGHT", "BACK", "FRONT"]
DLABELS = ["0-10", "10-20", "20-30", "30-40", "40-60", "60+"]

# Aggregate-CSV gate keys -> compact-table rows.
GATE_KEYS = [("distance_only", "Distance-only"),
             ("distance+density_p80", "Distance + density p80"),
             ("distance+density_p90", "Distance + density p90")]

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300,
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


# --------------------------------------------------------------------------- #
def load_aggregate():
    """Parse the pre-aggregated diagnostic stats into the K dict the figure /
    table builders consume. Every value is a column read from the CSV."""
    df = pd.read_csv(AGG)
    sec = lambda name: df[df["section"] == name]

    # --- panel (a): per-distance-bin mean support + zero-LiDAR rate ---
    b = sec("P1_distance_bins").set_index("distance_bin_m")
    bins = pd.DataFrame({
        "bin": DLABELS,
        "mean_pts": [float(b.loc[lab, "mean_pts"]) for lab in DLABELS],
        "zero_pct": [float(b.loc[lab, "pct_zero_pts"]) for lab in DLABELS],
        "n": [int(b.loc[lab, "n_rows"]) for lab in DLABELS],
    })

    # --- panel (b, top): near-range mean support by category ---
    c = sec("P2_by_category_near")
    cat = pd.DataFrame({"cat": c["cat"].values,
                        "mean": c["mean_pts"].astype(float).values,
                        "count": c["count"].astype(float).values})

    # --- panel (b, bottom): near-range mean support by camera view ---
    cam_df = sec("P2_by_camera_near")
    cam = pd.Series(cam_df["mean_pts"].astype(float).values,
                    index=cam_df["cam"].values)

    # --- main table: distance-only / p80 / p90 gates ---
    g = sec("P3_density_gate").set_index("gate")
    gates = {}
    for key, label in GATE_KEYS:
        r = g.loc[key]
        gates[key] = {
            "label": label,
            "retained_pct": float(r["pct_of_eligible"]),
            "mean_pts": float(r["mean_pts"]),
            "zero_pct": float(r["pct_zero_pts"]),
        }
    return {"bins": bins, "cat": cat, "cam": cam, "gates": gates}


# --------------------------------------------------------------------------- #
def make_figure(K):
    fig = plt.figure(figsize=(13.0, 5.8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1, 1],
                           hspace=0.62, wspace=0.30)
    ax_a = fig.add_subplot(gs[:, 0])     # panel (a): distance bins (tall)
    ax_b1 = fig.add_subplot(gs[0, 1])    # panel (b): by category
    ax_b2 = fig.add_subplot(gs[1, 1])    # panel (b): by camera view

    # ---- (a) mean in-box LiDAR points + zero-LiDAR % by distance bin ----
    b = K["bins"]
    x = np.arange(len(b))
    ax_a.bar(x, b["mean_pts"], color="#2c7fb8", edgecolor="black",
             linewidth=0.5, label="mean in-box LiDAR points")
    ax_a.set_xticks(x); ax_a.set_xticklabels(b["bin"])
    ax_a.set_xlabel("ego-centric distance bin (m)")
    ax_a.set_ylabel("mean in-box LiDAR points", color="#2c7fb8")
    ax_a.tick_params(axis="y", labelcolor="#2c7fb8")
    ax_a.set_ylim(0, b["mean_pts"].max() * 1.18)
    ax_a.set_title("(a) LiDAR support falls with distance;\nnear bins still show zero-LiDAR cases")
    ax_a.grid(axis="y", alpha=0.25)
    for xi, v in zip(x, b["mean_pts"]):
        ax_a.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8.5,
                  color="#08519c", weight="bold")
    axT = ax_a.twinx()
    axT.plot(x, b["zero_pct"], color="#e6550d", marker="o", lw=2,
             label="% zero-LiDAR")
    axT.set_ylabel("% zero-LiDAR", color="#e6550d")
    axT.tick_params(axis="y", labelcolor="#e6550d")
    axT.set_ylim(0, b["zero_pct"].max() * 1.25)
    for xi, v in zip(x, b["zero_pct"]):
        axT.text(xi, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8,
                 color="#a63603")
    h1, l1 = ax_a.get_legend_handles_labels()
    h2, l2 = axT.get_legend_handles_labels()
    ax_a.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right", framealpha=0.9)

    # ---- (b, top) within-30 m mean support by category ----
    cat = K["cat"].set_index("cat").reindex(CAT_SHOW).dropna()
    y = np.arange(len(cat))
    ax_b1.barh(y, cat["mean"], color="#3182bd", edgecolor="black", linewidth=0.4)
    ax_b1.set_yticks(y); ax_b1.set_yticklabels(cat.index, fontsize=8)
    ax_b1.invert_yaxis()
    ax_b1.set_xlabel("mean in-box LiDAR points (d $\\leq$ 30 m)")
    ax_b1.set_title("(b) Near-range support is object- and view-dependent",
                    fontsize=10.5)
    ax_b1.grid(axis="x", alpha=0.25)
    ax_b1.set_xlim(0, cat["mean"].max() * 1.13)
    for yi, v in zip(y, cat["mean"]):
        ax_b1.text(v, yi, f" {v:.0f}", va="center", fontsize=8)

    # ---- (b, bottom) within-30 m mean support by camera view ----
    cam = K["cam"].reindex(CAM_ORDER)
    xx = np.arange(len(cam))
    ax_b2.bar(xx, cam.values, color="#54a0c8", edgecolor="black", linewidth=0.4)
    ax_b2.set_xticks(xx)
    ax_b2.set_xticklabels([c.replace("_", "\n") for c in cam.index], fontsize=7.5)
    ax_b2.set_ylabel("mean pts (d $\\leq$ 30 m)", fontsize=8.5)
    ax_b2.grid(axis="y", alpha=0.25)
    ax_b2.set_ylim(0, cam.max() * 1.20)
    for xi, v in zip(xx, cam.values):
        ax_b2.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)

    for ext in ("pdf", "png"):
        out = os.path.join(FIG_OUT, f"fig_diagnostic_cl_redundancy.{ext}")
        fig.savefig(out, bbox_inches="tight"); print("wrote", out)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def make_tables(K):
    g = K["gates"]
    order = [k for k, _ in GATE_KEYS]

    # ---- machine-readable main table ----
    rows = [{
        "selection_rule": g[k]["label"],
        "near_range_rows_retained_pct": round(g[k]["retained_pct"], 1),
        "mean_in_box_lidar_points": round(g[k]["mean_pts"], 1),
        "zero_lidar_rate_pct": round(g[k]["zero_pct"], 1),
    } for k in order]
    out_csv = os.path.join(TAB_OUT, "table_cl_redundancy_main.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("wrote", out_csv)

    # ---- compact main-paper LaTeX table ----
    L = [
        "% Compact main-paper table. Requires \\usepackage{booktabs}.",
        "% Diagnostic analysis on nuScenes Part 1 object--camera projection rows;",
        "% NOT a model-performance evaluation. p80/p90 are percentile gates on",
        "% rho = n / A_2D (in-box LiDAR points / projected 2D box area), not raw",
        "% point-count thresholds. Generated by "
        "scripts/make_diagnostic_figure_and_table.py.",
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Distance--density pruning diagnostic on nuScenes Part 1. "
        "A distance-only candidate pool retains all near-range "
        "($d\\le30$\\,m) objects but is diluted by near-but-unsupported cases; "
        "an object-level density gate on $\\rho=n/A_{\\mathrm{2D}}$ (percentile "
        "thresholds) keeps the better-supported objects and removes the "
        "zero-LiDAR ones. Diagnostic analysis, not a model-performance "
        "evaluation.}",
        "  \\label{tab:cl-redundancy-diagnostic}",
        "  \\begin{tabular}{lrrr}",
        "    \\toprule",
        "    Selection rule & Near-range rows & Mean in-box & Zero-LiDAR \\\\",
        "                    & retained        & LiDAR points & rate \\\\",
        "    \\midrule",
    ]
    for k in order:
        r = g[k]
        L.append(f"    {r['label']} & {r['retained_pct']:.0f}\\% & "
                 f"{r['mean_pts']:.0f} & {r['zero_pct']:.1f}\\% \\\\")
    L += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]
    out_tex = os.path.join(TAB_OUT, "table_cl_redundancy_main.tex")
    with open(out_tex, "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote", out_tex)


# --------------------------------------------------------------------------- #
def main():
    os.makedirs(FIG_OUT, exist_ok=True)
    os.makedirs(TAB_OUT, exist_ok=True)
    K = load_aggregate()
    make_figure(K)
    make_tables(K)
    # console echo
    for k in [key for key, _ in GATE_KEYS]:
        r = K["gates"][k]
        print(f"  {r['label']:24s} retained {r['retained_pct']:5.1f}%  "
              f"mean {r['mean_pts']:6.1f}  zero {r['zero_pct']:.1f}%")


if __name__ == "__main__":
    main()
