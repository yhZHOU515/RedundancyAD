# ---------------------------------------------------------------------------
# Provenance: diagnostics/distance_density_diagnostic.py
# Role: Diagnostic stage 2: aggregate the projection table into data/results/diagnostic_aggregate_stats.csv (distance bins, quadrants, per-category / per-camera support, density-gate thresholds).
#
# REFERENCE / PROVENANCE SCRIPT -- documents how the shipped data/results CSVs
# were produced. NOT runnable from this package alone: it requires the
# excluded raw data (nuScenes), model checkpoints, framework source trees
# (MMDetection3D / BEVFusion, YOLO-LiDAR-Fusion), and the shared geometry/eval
# libraries from the full working tree (lib.py / lib3.py / holdout_lib.py /
# calibration.py). Absolute paths appear as /path/to/... placeholders.
# ---------------------------------------------------------------------------

"""
distance_density_diagnostic.py

Diagnostic evidence for the camera--LiDAR redundancy pruning rule used in the
journal paper. Tests the claim:

    "Distance is informative for camera--LiDAR redundancy, but distance alone is
     not sufficient. Even among near-range objects, LiDAR support varies
     substantially across objects, categories, and camera views. Therefore,
     camera--LiDAR redundancy candidates should be identified using both
     geometric proximity AND object-level LiDAR support/density."

This is DIAGNOSTIC ONLY. It does not train, evaluate, or modify the
pruning/evaluation pipeline. It reads the existing per-object table produced by
the nuScenes camera--LiDAR projection step and writes clean figures/tables.

Input
-----
camera_lidar_exploratory_stats_all_cameras_part1.csv
  nuScenes Part 1, scene-level 80/20 split source table; one row per
  (object annotation, camera view) where the 3D box projects into that camera.
  Columns used:
    distance_m        ego-centric distance d to the 3D box centre (m)
    lidar_point_count in-box LiDAR point count n (points inside the 3D box)
    category_name     nuScenes object category
    camera_name       camera view the box projects into
    bbox_x1..y2       projected 2D box (pixels) -> 2D area A_2D
    image_width/height

Derived per object
------------------
    d      = distance_m
    n      = lidar_point_count
    A_2D   = (bbox_x2-bbox_x1) * (bbox_y2-bbox_y1)         [px^2]
    rho    = n / A_2D                                       [points / px^2]
           reported as points per 1000 px^2 for readability (rho_k = rho*1000)

Conventions (match the existing pipeline)
-----------------------------------------
    NEAR cutoff       d <= 30 m
    SUPPORTED (pts)   n >= 10            (in-box LiDAR points; pipeline default)

Outputs (analysis/distance_density_diagnostic/)
-----------------------------------------------
  Part 1  fig1_distance_vs_support.{png,pdf}      scatter + binned trend
          quadrant_counts.csv
  Part 2  fig2_heterogeneity_by_category.{png,pdf}
          fig3_heterogeneity_by_camera.{png,pdf}
          heterogeneity_by_category.csv
          heterogeneity_by_camera.csv
  Part 3  fig4_density_gate.{png,pdf}
          density_gate_thresholds.csv
  Part 4  diagnostic_summary.csv            (one consolidated stats table)
          README_diagnostic.md              (short report)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import kruskal, spearmanr
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
# Clean-package layout: the large source table is NOT shipped (see README).
# Point CL_DIAGNOSTIC_CSV at your local copy of the per-(object,camera) table.
CSV = os.environ.get(
    "CL_DIAGNOSTIC_CSV",
    os.path.join(HERE, "inputs", "camera_lidar_exploratory_stats_all_cameras_part1.csv"))
OUT = os.path.join(HERE, "outputs")

NEAR_M = 30.0          # near-distance cutoff (m)
SUPPORT_PTS = 10       # "supported" = in-box LiDAR points >= this (pipeline default)
DENSITY_PCTLS = [50, 60, 70, 80, 90]      # density-gate percentiles (primary)
RAW_PT_THRESH = [5, 10, 20, 30]           # raw point-count gates (exploratory)

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
})


def short_cat(name):
    """Compact nuScenes category label for plots/tables."""
    return name.replace("vehicle.", "").replace("human.pedestrian.", "ped.") \
               .replace("movable_object.", "").replace("static_object.", "")


def short_cam(name):
    return name.replace("CAM_", "")


def savefig(fig, stem):
    fig.savefig(os.path.join(OUT, stem + ".png"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, stem + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def fmt_p(p):
    """Readable p-value; avoid the misleading '0.0e+00' on float underflow."""
    return "<1e-300" if p == 0 else f"{p:.1e}"


def grp_stats(series):
    """mean/median/count/zero-% for a value series."""
    n = len(series)
    return {
        "count": int(n),
        "mean": float(series.mean()) if n else np.nan,
        "median": float(series.median()) if n else np.nan,
        "pct_zero": 100.0 * float((series == 0).mean()) if n else np.nan,
    }


# --------------------------------------------------------------------------- #
def load():
    df = pd.read_csv(CSV)
    df["d"] = df["distance_m"].astype(float)
    df["n"] = df["lidar_point_count"].astype(float)
    df["A_2D"] = (df["bbox_x2"] - df["bbox_x1"]) * (df["bbox_y2"] - df["bbox_y1"])
    df = df[df["A_2D"] > 0].copy()
    df["rho"] = df["n"] / df["A_2D"]              # points / px^2
    df["rho_k"] = df["rho"] * 1000.0             # points / 1000 px^2
    df["cat"] = df["category_name"].map(short_cat)
    df["cam"] = df["camera_name"].map(short_cam)
    df["near"] = df["d"] <= NEAR_M
    df["supported"] = df["n"] >= SUPPORT_PTS
    return df


# --------------------------------------------------------------------------- #
# PART 1 — distance vs LiDAR support; quadrant counts
# --------------------------------------------------------------------------- #
def part1(df):
    EDGES = [0, 10, 20, 30, 40, 60, np.inf]
    LABELS = ["0-10", "10-20", "20-30", "30-40", "40-60", "60+"]
    df["dbin"] = pd.cut(df["d"], bins=EDGES, labels=LABELS, right=True,
                        include_lowest=True)

    binrows = []
    for lab in LABELS:
        g = df[df["dbin"] == lab]
        s = grp_stats(g["n"])
        binrows.append({
            "distance_bin_m": lab, "n_rows": s["count"],
            "mean_pts": s["mean"], "median_pts": s["median"],
            "pct_zero_pts": s["pct_zero"],
            "pct_supported": 100.0 * float((g["n"] >= SUPPORT_PTS).mean()) if s["count"] else np.nan,
            "mean_rho_k": float(g["rho_k"].mean()) if s["count"] else np.nan,
            "median_rho_k": float(g["rho_k"].median()) if s["count"] else np.nan,
        })
    bintab = pd.DataFrame(binrows)

    # Quadrants: near/far (30 m) x dense/sparse (>=SUPPORT_PTS in-box points)
    near, far = df["near"], ~df["near"]
    dense = df["supported"]
    quad = {
        "near_dense": int((near & dense).sum()),
        "near_sparse": int((near & ~dense).sum()),
        "far_dense": int((far & dense).sum()),
        "far_sparse": int((far & ~dense).sum()),
    }
    tot = len(df)
    quad_df = pd.DataFrame([
        {"quadrant": k, "definition": d, "count": v,
         "pct_of_all": round(100.0 * v / tot, 2)}
        for (k, v), d in zip(quad.items(), [
            f"d<={NEAR_M:.0f}m & pts>={SUPPORT_PTS}",
            f"d<={NEAR_M:.0f}m & pts<{SUPPORT_PTS}",
            f"d>{NEAR_M:.0f}m & pts>={SUPPORT_PTS}",
            f"d>{NEAR_M:.0f}m & pts<{SUPPORT_PTS}"])
    ])
    quad_df.to_csv(os.path.join(OUT, "quadrant_counts.csv"), index=False)
    bintab.to_csv(os.path.join(OUT, "distance_bins.csv"), index=False)

    # ---- Figure 1: scatter (left) + binned trend (right) ----
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # Left: distance vs in-box LiDAR points (log y), subsampled for clarity.
    rng = np.random.RandomState(0)
    idx = rng.choice(len(df), size=min(25000, len(df)), replace=False)
    sub = df.iloc[idx]
    yplot = sub["n"].clip(lower=0.5)  # 0 -> 0.5 so it shows on log axis
    axL.scatter(sub["d"], yplot, s=4, alpha=0.10, color="#2c7fb8",
                edgecolors="none")
    axL.set_yscale("log")
    axL.axvline(NEAR_M, color="black", ls="--", lw=1.2)
    axL.axhline(SUPPORT_PTS, color="#e6550d", ls="--", lw=1.2)
    axL.text(NEAR_M + 1, axL.get_ylim()[1] * 0.5, f"near cutoff {NEAR_M:.0f} m",
             fontsize=8, rotation=90, va="top")
    axL.text(160, SUPPORT_PTS * 1.25, f"support {SUPPORT_PTS} pts",
             fontsize=8, color="#e6550d")
    axL.set_xlabel("Ego-centric distance d (m)")
    axL.set_ylabel("In-box LiDAR points n (log; 0 shown at 0.5)")
    axL.set_title("LiDAR support vs distance (per object-camera)")
    axL.set_xlim(0, 120)
    axL.grid(alpha=0.25)
    # quadrant annotations
    axL.text(8, axL.get_ylim()[1] * 0.4, f"near-dense\n{quad['near_dense']:,}",
             fontsize=8, ha="center", color="#08519c", weight="bold")
    axL.text(8, 0.7, f"near-sparse\n{quad['near_sparse']:,}",
             fontsize=8, ha="center", color="#a63603", weight="bold")

    # Right: binned mean points + % zero on twin axis
    x = np.arange(len(LABELS))
    axR.bar(x, bintab["mean_pts"], color="#2c7fb8", edgecolor="black",
            linewidth=0.5, label="mean in-box LiDAR pts")
    axR.set_xticks(x); axR.set_xticklabels(LABELS)
    axR.set_xlabel("Distance bin (m)")
    axR.set_ylabel("Mean in-box LiDAR points", color="#2c7fb8")
    axR.tick_params(axis="y", labelcolor="#2c7fb8")
    axR.set_title("Support falls with distance; near bins still have zeros")
    axR.grid(axis="y", alpha=0.25)
    for xi, v in zip(x, bintab["mean_pts"]):
        axR.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    axT = axR.twinx()
    axT.plot(x, bintab["pct_zero_pts"], color="#e6550d", marker="o", lw=2,
             label="% zero-LiDAR")
    axT.set_ylabel("% rows with zero LiDAR points", color="#e6550d")
    axT.tick_params(axis="y", labelcolor="#e6550d")
    axT.set_ylim(0, max(bintab["pct_zero_pts"]) * 1.25)
    for xi, v in zip(x, bintab["pct_zero_pts"]):
        axT.text(xi, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=8,
                 color="#e6550d")

    fig.suptitle("Part 1 — Distance is informative but does not guarantee LiDAR "
                 "support (nuScenes Part 1)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, "fig1_distance_vs_support")

    return bintab, quad_df, quad


# --------------------------------------------------------------------------- #
# PART 2 — distance-controlled heterogeneity (d <= NEAR_M)
# --------------------------------------------------------------------------- #
def part2(df):
    near = df[df["near"]].copy()

    def by(col, min_count):
        rows = []
        for key, g in near.groupby(col):
            if len(g) < min_count:
                continue
            sp = grp_stats(g["n"])     # support = in-box points
            de = grp_stats(g["rho_k"])  # density (points/1000px^2)
            rows.append({
                col: key, "count": sp["count"],
                "mean_pts": sp["mean"], "median_pts": sp["median"],
                "pct_zero_pts": sp["pct_zero"],
                "pct_supported": 100.0 * float((g["n"] >= SUPPORT_PTS).mean()),
                "mean_rho_k": de["mean"], "median_rho_k": de["median"],
            })
        t = pd.DataFrame(rows).sort_values("mean_pts", ascending=False)
        return t

    # keep categories with a meaningful near-range sample
    cat_tab = by("cat", min_count=100)
    cam_tab = by("cam", min_count=1)

    # Kruskal-Wallis: does support vary across groups after fixing distance?
    kw = {}
    if HAVE_SCIPY:
        cats = [g["n"].values for _, g in near.groupby("cat") if len(g) >= 100]
        cams = [g["n"].values for _, g in near.groupby("cam")]
        if len(cats) > 1:
            H, p = kruskal(*cats)
            kw["category"] = {"H": float(H), "p": float(p), "k_groups": len(cats)}
        if len(cams) > 1:
            H, p = kruskal(*cams)
            kw["camera"] = {"H": float(H), "p": float(p), "k_groups": len(cams)}

    cat_tab.to_csv(os.path.join(OUT, "heterogeneity_by_category.csv"), index=False)
    cam_tab.to_csv(os.path.join(OUT, "heterogeneity_by_camera.csv"), index=False)

    # ---- Figure 2: by category ----
    t = cat_tab.copy()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.6))
    y = np.arange(len(t))
    a1.barh(y, t["mean_pts"], color="#3182bd", edgecolor="black", linewidth=0.4)
    a1.set_yticks(y); a1.set_yticklabels(t["cat"]); a1.invert_yaxis()
    a1.set_xlabel("Mean in-box LiDAR points")
    a1.set_title("Mean LiDAR support by category (near range, d<=30 m)")
    a1.grid(axis="x", alpha=0.25)
    for yi, v, c in zip(y, t["mean_pts"], t["count"]):
        a1.text(v, yi, f" {v:.0f} (n={c:,})", va="center", fontsize=7.5)

    a2.barh(y, t["pct_zero_pts"], color="#e6550d", edgecolor="black", linewidth=0.4)
    a2.set_yticks(y); a2.set_yticklabels(t["cat"]); a2.invert_yaxis()
    a2.set_xlabel("% with zero in-box LiDAR points")
    a2.set_title("Zero-LiDAR rate by category (near range)")
    a2.grid(axis="x", alpha=0.25)
    for yi, v in zip(y, t["pct_zero_pts"]):
        a2.text(v, yi, f" {v:.1f}%", va="center", fontsize=7.5)
    sub = ("Part 2 — After fixing distance (<=30 m), LiDAR support still varies "
           "by category")
    if "category" in kw:
        sub += f"  (Kruskal-Wallis H={kw['category']['H']:.0f}, p={fmt_p(kw['category']['p'])})"
    fig.suptitle(sub, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    savefig(fig, "fig2_heterogeneity_by_category")

    # ---- Figure 3: by camera ----
    t = cam_tab.copy()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    x = np.arange(len(t))
    a1.bar(x, t["mean_pts"], color="#3182bd", edgecolor="black", linewidth=0.4)
    a1.set_xticks(x); a1.set_xticklabels(t["cam"], rotation=30, ha="right")
    a1.set_ylabel("Mean in-box LiDAR points")
    a1.set_title("Mean LiDAR support by camera view (d<=30 m)")
    a1.grid(axis="y", alpha=0.25)
    for xi, v, c in zip(x, t["mean_pts"], t["count"]):
        a1.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)

    a2.bar(x, t["pct_zero_pts"], color="#e6550d", edgecolor="black", linewidth=0.4)
    a2.set_xticks(x); a2.set_xticklabels(t["cam"], rotation=30, ha="right")
    a2.set_ylabel("% with zero in-box LiDAR points")
    a2.set_title("Zero-LiDAR rate by camera view (d<=30 m)")
    a2.grid(axis="y", alpha=0.25)
    for xi, v in zip(x, t["pct_zero_pts"]):
        a2.text(xi, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    sub = "Part 3 — Near-range LiDAR support is also view-dependent"
    if "camera" in kw:
        sub += f"  (Kruskal-Wallis H={kw['camera']['H']:.0f}, p={fmt_p(kw['camera']['p'])})"
    fig.suptitle(sub, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    savefig(fig, "fig3_heterogeneity_by_camera")

    return cat_tab, cam_tab, kw


# --------------------------------------------------------------------------- #
# PART 3 — density-gate motivation
# --------------------------------------------------------------------------- #
def part3(df):
    """Compare distance-only vs distance+density candidate pools.

    Eligible pool = near objects (d <= NEAR_M). Distance-only selects ALL of
    them. A density/support gate keeps only the better-supported near objects.
    """
    elig = df[df["near"]].copy()
    n_elig = len(elig)

    rows = []
    # distance-only baseline
    s = grp_stats(elig["n"])
    rows.append({
        "gate": "distance_only", "type": "baseline", "threshold": "-",
        "n_selected": n_elig, "pct_of_eligible": 100.0,
        "mean_pts": s["mean"], "median_pts": s["median"],
        "pct_zero_pts": s["pct_zero"],
    })

    # density percentile gates (primary): rho >= p-th percentile of eligible rho
    for p in DENSITY_PCTLS:
        thr = float(np.percentile(elig["rho_k"], p))
        sel = elig[elig["rho_k"] >= thr]
        s = grp_stats(sel["n"])
        rows.append({
            "gate": f"distance+density_p{p}", "type": "density_percentile",
            "threshold": f"rho_k>={thr:.4f} (p{p})",
            "n_selected": len(sel),
            "pct_of_eligible": 100.0 * len(sel) / n_elig,
            "mean_pts": s["mean"], "median_pts": s["median"],
            "pct_zero_pts": s["pct_zero"],
        })

    # raw point-count gates (exploratory)
    for t in RAW_PT_THRESH:
        sel = elig[elig["n"] >= t]
        s = grp_stats(sel["n"])
        rows.append({
            "gate": f"distance+points>={t}", "type": "raw_points_exploratory",
            "threshold": f"n>={t}",
            "n_selected": len(sel),
            "pct_of_eligible": 100.0 * len(sel) / n_elig,
            "mean_pts": s["mean"], "median_pts": s["median"],
            "pct_zero_pts": s["pct_zero"],
        })

    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(OUT, "density_gate_thresholds.csv"), index=False)

    # ---- Figure 4: pool shrink vs purity ----
    dens = tab[tab["type"].isin(["baseline", "density_percentile"])].copy()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.0))
    labels = ["dist-only"] + [f"p{p}" for p in DENSITY_PCTLS]
    x = np.arange(len(labels))

    a1.bar(x, dens["pct_of_eligible"], color="#756bb1", edgecolor="black",
           linewidth=0.4)
    a1.set_xticks(x); a1.set_xticklabels(labels)
    a1.set_xlabel("Density gate (percentile of near-range density)")
    a1.set_ylabel("% of eligible near objects selected")
    a1.set_title("A density gate shrinks the candidate pool")
    a1.grid(axis="y", alpha=0.25)
    for xi, v in zip(x, dens["pct_of_eligible"]):
        a1.text(xi, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=8)

    a2.plot(x, dens["pct_zero_pts"], color="#e6550d", marker="o", lw=2,
            label="% zero-LiDAR in pool")
    a2.set_xticks(x); a2.set_xticklabels(labels)
    a2.set_xlabel("Density gate (percentile)")
    a2.set_ylabel("% zero-LiDAR in selected pool", color="#e6550d")
    a2.tick_params(axis="y", labelcolor="#e6550d")
    a2.set_title("...and removes near-but-unsupported objects")
    a2.grid(alpha=0.25)
    for xi, v in zip(x, dens["pct_zero_pts"]):
        a2.text(xi, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8,
                color="#e6550d")
    aT = a2.twinx()
    aT.plot(x, dens["mean_pts"], color="#2c7fb8", marker="s", lw=2,
            label="mean pts in pool")
    aT.set_ylabel("mean in-box LiDAR points in pool", color="#2c7fb8")
    aT.tick_params(axis="y", labelcolor="#2c7fb8")

    fig.suptitle("Part 3 — Density gate motivation: distance-only pool is diluted "
                 "by near-but-unsupported objects", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    savefig(fig, "fig4_density_gate")

    return tab, n_elig


# --------------------------------------------------------------------------- #
# PART 4 — consolidated CSV + Markdown report
# --------------------------------------------------------------------------- #
def part4(df, bintab, quad, cat_tab, cam_tab, kw, gate_tab, n_elig):
    # consolidated stats CSV (long format, section-tagged)
    blocks = []
    b = bintab.copy(); b.insert(0, "section", "P1_distance_bins"); blocks.append(b)
    q = pd.read_csv(os.path.join(OUT, "quadrant_counts.csv"))
    q.insert(0, "section", "P1_quadrants"); blocks.append(q)
    c = cat_tab.copy(); c.insert(0, "section", "P2_by_category_near"); blocks.append(c)
    m = cam_tab.copy(); m.insert(0, "section", "P2_by_camera_near"); blocks.append(m)
    g = gate_tab.copy(); g.insert(0, "section", "P3_density_gate"); blocks.append(g)
    consolidated = pd.concat(blocks, ignore_index=True)
    consolidated.to_csv(os.path.join(OUT, "diagnostic_summary.csv"), index=False)

    # ---- numbers for the report ----
    N = len(df)
    near_mask = df["near"]
    n_near = int(near_mask.sum())
    near_zero_pct = 100.0 * float((df.loc[near_mask, "n"] == 0).mean())
    near_unsupported_pct = 100.0 * float((df.loc[near_mask, "n"] < SUPPORT_PTS).mean())
    mean_pts_0_10 = float(bintab.iloc[0]["mean_pts"])
    mean_pts_last = float(bintab.iloc[-1]["mean_pts"])
    zero_0_10 = float(bintab.iloc[0]["pct_zero_pts"])
    zero_last = float(bintab.iloc[-1]["pct_zero_pts"])

    # spearman d vs n (monotone trend)
    sp_txt = ""
    if HAVE_SCIPY:
        rho_s, p_s = spearmanr(df["d"], df["n"])
        sp_txt = (f"Spearman correlation between distance and in-box LiDAR "
                  f"points is rho={rho_s:+.3f} (p={fmt_p(p_s)}) — a clear but "
                  f"imperfect monotone decline.")

    cat_lo = cat_tab.iloc[-1]; cat_hi = cat_tab.iloc[0]
    cam_lo = cam_tab.sort_values("mean_pts").iloc[0]
    cam_hi = cam_tab.sort_values("mean_pts").iloc[-1]

    p80 = gate_tab[gate_tab["gate"] == "distance+density_p80"]
    p80 = p80.iloc[0] if len(p80) else None

    L = []
    L += ["# Camera–LiDAR redundancy diagnostic: distance is informative but insufficient",
          "",
          "**Scope.** Diagnostic evidence for the camera–LiDAR distance–density "
          "pruning rule. No training, evaluation, or pipeline changes — this reads "
          "the existing per-object projection table and produces figures/tables only.",
          "",
          "## What was computed",
          "",
          "For every (object annotation, camera view) row where a 3D box projects "
          "into a camera image, we computed:",
          "",
          "- ego-centric distance `d` (m),",
          "- in-box LiDAR point count `n` (points inside the 3D box),",
          "- projected 2D box area `A_2D` (px²) from the stored 2D box,",
          "- LiDAR support density `rho = n / A_2D` (reported as points / 1000 px²),",
          "- object category and associated camera view.",
          "",
          "We then (1) characterised support vs distance and counted "
          "near/far × dense/sparse quadrants; (2) restricted to near range "
          f"(`d ≤ {NEAR_M:.0f} m`) and compared support across categories and "
          "camera views; and (3) compared a distance-only candidate pool with "
          "distance+density pools at several gates.",
          "",
          "## Dataset / split / source",
          "",
          f"- **Source table:** `camera_lidar_exploratory_stats_all_cameras_part1.csv`",
          f"- **Dataset:** nuScenes, **Part 1** (scene-level 80/20 split source table; "
          "same projection step that feeds the BEVFusion/camera–LiDAR pruning pipeline).",
          f"- **Unit:** one row per (annotation, camera) projection.",
          f"- **Sample size:** **{N:,}** object–camera rows; **{n_near:,}** are near "
          f"range (`d ≤ {NEAR_M:.0f} m`).",
          f"- **Conventions (match pipeline):** near cutoff `d ≤ {NEAR_M:.0f} m`; "
          f"\"supported\" = `n ≥ {SUPPORT_PTS}` in-box LiDAR points.",
          "",
          "## Main numerical findings",
          "",
          "### 1. Distance is informative",
          "",
          f"- Mean in-box LiDAR points fall from **{mean_pts_0_10:.0f}** in the 0–10 m "
          f"bin to **{mean_pts_last:.0f}** in the 60 m+ bin; the zero-LiDAR rate rises "
          f"from **{zero_0_10:.1f}%** to **{zero_last:.1f}%**.",
          (f"- {sp_txt}" if sp_txt else ""),
          "",
          "### 2. ...but distance alone is insufficient (near range is not uniformly supported)",
          "",
          f"- Among the {n_near:,} near-range objects (`d ≤ {NEAR_M:.0f} m`), "
          f"**{near_zero_pct:.1f}%** have **zero** in-box LiDAR points and "
          f"**{near_unsupported_pct:.1f}%** fall below the support threshold "
          f"(`n < {SUPPORT_PTS}`).",
          "- Quadrant counts (near/far × dense/sparse):",
          "",
          "| quadrant | definition | count | % of all |",
          "|---|---|---|---|"]
    for _, r in q.iterrows():
        L.append(f"| {r['quadrant']} | {r['definition']} | {int(r['count']):,} | "
                 f"{r['pct_of_all']:.1f}% |")
    L += ["",
          "The **near-sparse** quadrant is exactly the set a distance-only rule "
          "would wrongly treat as LiDAR-redundant.",
          "",
          "### 3. After controlling for distance, support still varies by category and view",
          "",
          f"- **By category (near range):** mean in-box points range from "
          f"**{cat_hi['mean_pts']:.0f}** ({cat_hi['cat']}) down to "
          f"**{cat_lo['mean_pts']:.0f}** ({cat_lo['cat']}); zero-LiDAR rate ranges "
          f"from {cat_tab['pct_zero_pts'].min():.1f}% to "
          f"{cat_tab['pct_zero_pts'].max():.1f}%."]
    if "category" in kw:
        L.append(f"  Kruskal–Wallis across categories: H={kw['category']['H']:.0f}, "
                 f"p={fmt_p(kw['category']['p'])} ({kw['category']['k_groups']} groups).")
    L += [f"- **By camera view (near range):** mean in-box points range from "
          f"**{cam_hi['mean_pts']:.0f}** ({cam_hi['cam']}) down to "
          f"**{cam_lo['mean_pts']:.0f}** ({cam_lo['cam']})."]
    if "camera" in kw:
        L.append(f"  Kruskal–Wallis across views: H={kw['camera']['H']:.0f}, "
                 f"p={fmt_p(kw['camera']['p'])} ({kw['camera']['k_groups']} groups).")
    L += ["",
          "### 4. A density gate removes near-but-unsupported objects",
          "",
          "| gate | n selected | % of eligible | mean pts | median pts | % zero-LiDAR |",
          "|---|---|---|---|---|---|"]
    for _, r in gate_tab.iterrows():
        L.append(f"| {r['gate']} | {int(r['n_selected']):,} | "
                 f"{r['pct_of_eligible']:.1f}% | {r['mean_pts']:.1f} | "
                 f"{r['median_pts']:.0f} | {r['pct_zero_pts']:.1f}% |")
    L += ["",
          "(Density-percentile gates are primary; raw point-count gates are labelled "
          "exploratory.)",
          ""]
    if p80 is not None:
        L.append(f"- Adding a density gate at the **p80** density of near-range "
                 f"objects keeps {p80['pct_of_eligible']:.0f}% of the eligible pool "
                 f"while lifting mean support to {p80['mean_pts']:.0f} pts and "
                 f"cutting the zero-LiDAR fraction to {p80['pct_zero_pts']:.1f}%.")
    L += ["",
          "## Interpretation",
          "",
          "- **Distance is informative but insufficient.** LiDAR support declines "
          "strongly with distance, yet a substantial fraction of near-range objects "
          "carry weak or zero in-box LiDAR returns.",
          "- **Distance-only pruning may be overly aggressive:** it would label every "
          "near object as LiDAR-redundant, including the near-sparse objects above, "
          "for which the camera is likely the load-bearing sensor.",
          "- **Near-range objects can have weak LiDAR support** because of occlusion, "
          "grazing geometry, small or low-reflectivity surfaces, and "
          "view-dependent coverage.",
          "- **Camera–LiDAR redundancy is object- and view-dependent**, not a pure "
          "function of range — support varies significantly across categories and "
          "camera views even at fixed distance.",
          "- **Density provides object-level sensor-support evidence** that distance "
          "cannot: a density/support gate prunes the near-but-unsupported objects a "
          "distance-only rule would wrongly include.",
          "",
          "## Does the evidence support the claim?",
          "",
          "Yes. The data support the claim that **distance alone is insufficient** "
          "for identifying camera–LiDAR redundancy, and that **object-level LiDAR "
          "support/density should be added as a second gate** alongside geometric "
          "proximity. The thresholds used here "
          f"(`d ≤ {NEAR_M:.0f} m`, `n ≥ {SUPPORT_PTS}`, density percentiles) are "
          "data-informed operational choices for this nuScenes setup, not universal "
          "constants; they would be re-tuned for a different sensor suite or dataset. "
          "This is a redundancy/pruning argument, not a safety guarantee.",
          "",
          "## Files",
          "",
          "- `fig1_distance_vs_support.{png,pdf}` — scatter + binned support trend, quadrants",
          "- `fig2_heterogeneity_by_category.{png,pdf}` — near-range support by category",
          "- `fig3_heterogeneity_by_camera.{png,pdf}` — near-range support by camera view",
          "- `fig4_density_gate.{png,pdf}` — density-gate pool shrink vs purity",
          "- `diagnostic_summary.csv` — consolidated stats (all sections)",
          "- `distance_bins.csv`, `quadrant_counts.csv`, "
          "`heterogeneity_by_category.csv`, `heterogeneity_by_camera.csv`, "
          "`density_gate_thresholds.csv` — per-section tables",
          ""]
    L = [x for x in L if x is not None]
    with open(os.path.join(OUT, "README_diagnostic.md"), "w") as f:
        f.write("\n".join(L) + "\n")


# --------------------------------------------------------------------------- #
def main():
    os.makedirs(OUT, exist_ok=True)
    df = load()
    print(f"Loaded {len(df):,} object-camera rows from {os.path.basename(CSV)}")
    bintab, quad_df, quad = part1(df)
    cat_tab, cam_tab, kw = part2(df)
    gate_tab, n_elig = part3(df)
    part4(df, bintab, quad, cat_tab, cam_tab, kw, gate_tab, n_elig)
    print("\nQuadrants:", quad)
    print("\nDensity-gate table:")
    print(gate_tab.to_string(index=False))
    print(f"\nAll outputs in {OUT}/")


if __name__ == "__main__":
    main()
