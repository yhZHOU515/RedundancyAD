"""Approximately matched deduplicated-point-budget table (p90 gate).

Reads   data/point_budget_match_p90.csv
Writes  outputs/tables/table_point_budget_match_p90.{csv,tex}

This is the second way of equalising how much is removed. Instead of matching the
number of selected boxes, the distance-only threshold is lowered until the
cumulative number of removed deduplicated keyframe LiDAR points is closest to the
number removed by the p90 gate. The selected-box count is then an outcome rather
than a target, which is why box count and removed-point volume must not be treated
as the same budget.

Both arms are evaluated against the same saved full-sensor baseline, so their
lost-ratios are directly comparable.

Run:  python scripts/make_point_budget_table.py
"""
import os
from _common import DISTANCES, TAB_OUT, fmt_d, load, write_csv, write_tex

ARMS = [("distance_density_p90", "distance+density p90"),
        ("distance_only_point_matched", "distance-only, point matched")]


def main():
    rows = load("point_budget_match_p90.csv")
    by = {(float(r["T_dist_m"]), r["arm"]): r for r in rows}

    header = ["T_dist_m", "arm", "distance_cutoff_m", "target_dedup_points",
              "achieved_dedup_points", "point_budget_mismatch_pct", "selected_box_count",
              "lost_ratio", "mAP50", "native_mAP", "baseline_condition_id"]
    out = []
    for d in DISTANCES:
        for arm, _ in ARMS:
            r = by[(d, arm)]
            out.append([fmt_d(d), arm, "%.2f" % float(r["distance_cutoff_m"]),
                        r["target_dedup_points"], r["achieved_dedup_points"],
                        "%+.3f" % (100.0 * float(r["rel_point_mismatch"])),
                        r["selected_box_count"], "%.5f" % float(r["lost_ratio"]),
                        "%.5f" % float(r["mAP50"]), "%.5f" % float(r["native_mAP"]),
                        r["baseline_condition_id"]])
    write_csv(os.path.join(TAB_OUT, "table_point_budget_match_p90.csv"), header, out)

    worst = max(abs(100.0 * float(r["rel_point_mismatch"])) for r in rows)
    lower = sum(1 for d in DISTANCES
                if float(by[(d, "distance_only_point_matched")]["lost_ratio"])
                < float(by[(d, "distance_density_p90")]["lost_ratio"]))
    tex = ["% Auto-generated from data/point_budget_match_p90.csv by "
           "scripts/make_point_budget_table.py",
           r"\begin{table}[t]", r"  \centering",
           "  \\caption{Approximately matched deduplicated-point budget for the p90 gate. "
           "The distance-only threshold is lowered until the removed keyframe LiDAR points "
           "are closest to the p90 budget; the selected-box count is an outcome, not a "
           "target. Every setting matches the budget to within %.2f\\%%. The point-matched "
           "distance-only comparator has the lower lost-ratio at %d of %d thresholds.}"
           % (worst, lower, len(DISTANCES)),
           r"  \label{tab:point-budget-match-p90}",
           r"  \begin{tabular}{llrrcc}", r"    \toprule",
           r"    $T_{\mathrm{dist}}$ & Arm & Points & Boxes & Lost-ratio & mAP@0.5 \\",
           r"    \midrule"]
    for i, d in enumerate(DISTANCES):
        if i:
            tex.append(r"    \midrule")
        for j, (arm, label) in enumerate(ARMS):
            r = by[(d, arm)]
            lab = (r"    \multirow{2}{*}{%s\,m}" % fmt_d(d)) if j == 0 else "    "
            tex.append("%s & %s & %s & %s & %.5f & %.3f \\\\" % (
                lab, label,
                "{:,}".format(int(r["achieved_dedup_points"])).replace(",", "{,}"),
                r["selected_box_count"], float(r["lost_ratio"]), float(r["mAP50"])))
    tex += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    write_tex(os.path.join(TAB_OUT, "table_point_budget_match_p90.tex"), tex)
    print(f"  point-budget worst mismatch {worst:.3f}%; "
          f"point-matched distance-only lower lost-ratio at {lower}/{len(DISTANCES)} thresholds")


if __name__ == "__main__":
    main()
