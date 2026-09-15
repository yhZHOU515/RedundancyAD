"""Matched selected-box-count tables (reviewer-requested comparison).

Reads   data/matched_box_count_pairs.csv          (20 settings)
Writes  outputs/tables/table_matched_box_count_lost_ratio.{csv,tex}
        outputs/tables/table_matched_box_count_metric_summary.{csv,tex}

For each (T_dist, density percentile) setting the distance-only threshold was
lowered until it selected the same number of boxes as the density gate. Values
are read verbatim from the shipped CSV; nothing is recomputed from raw data.

Run:  python scripts/make_matched_box_count_tables.py
"""
import os
from _common import DISTANCES, PERCENTILES, TAB_OUT, fmt_d, load, write_csv, write_tex


def main():
    rows = load("matched_box_count_pairs.csv")
    by = {(float(r["D"]), int(r["percentile"])): r for r in rows}

    # ---- per-setting lost-ratio table ----------------------------------
    header = ["T_dist_m", "density_percentile", "matched_selected_boxes",
              "distance_only_effective_cutoff_m", "density_lost_ratio",
              "distance_only_lost_ratio", "delta_lost_ratio_density_minus_distance_only"]
    out = []
    for d in DISTANCES:
        for p in PERCENTILES:
            r = by[(d, p)]
            out.append([fmt_d(d), p, r["matched_N"],
                        "%.2f" % float(r["distance_only_effective_cutoff"]),
                        "%.5f" % float(r["density_lost_ratio"]),
                        "%.5f" % float(r["do_lost_ratio"]),
                        "%+.5f" % float(r["delta_lost_ratio"])])
    write_csv(os.path.join(TAB_OUT, "table_matched_box_count_lost_ratio.csv"), header, out)

    tex = ["% Auto-generated from data/matched_box_count_pairs.csv by "
           "scripts/make_matched_box_count_tables.py",
           r"\begin{table}[t]", r"  \centering",
           r"  \caption{Matched selected-box-count comparison on the BEVFusion 25-scene "
           r"nuScenes holdout. For each setting the distance-only threshold is lowered "
           r"until it selects the same number of boxes as the density gate. Lost-ratio is "
           r"at IoU $\geq 0.5$ (lower is better); $\Delta$ is density minus distance-only.}",
           r"  \label{tab:matched-box-count-holdout}",
           r"  \begin{tabular}{llrrccc}", r"    \toprule",
           r"    $T_{\mathrm{dist}}$ & $T_{\rho}$ & Boxes & Cutoff (m) & "
           r"Density & Distance-only & $\Delta$ \\", r"    \midrule"]
    for i, d in enumerate(DISTANCES):
        if i:
            tex.append(r"    \midrule")
        for j, p in enumerate(PERCENTILES):
            r = by[(d, p)]
            label = (r"    \multirow{4}{*}{%s\,m}" % fmt_d(d)) if j == 0 else "    "
            tex.append("%s & p%d & %s & %.2f & %.5f & %.5f & $%+.5f$ \\\\" % (
                label, p, r["matched_N"], float(r["distance_only_effective_cutoff"]),
                float(r["density_lost_ratio"]), float(r["do_lost_ratio"]),
                float(r["delta_lost_ratio"])))
    tex += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    write_tex(os.path.join(TAB_OUT, "table_matched_box_count_lost_ratio.tex"), tex)

    # ---- metric-direction summary --------------------------------------
    metrics = [("Lost-ratio (lower is better)", "density_lost_ratio", "do_lost_ratio", True),
               ("mAP@0.5 (higher is better)", "density_mAP50", "do_mAP50", False),
               ("Native nuScenes mAP (higher is better)", "density_native_mAP",
                "do_native_mAP", False)]
    header = ["metric", "settings_favouring_distance_only", "settings_favouring_density",
              "settings_tied", "total_settings"]
    out = []
    for name, dcol, ocol, lower_better in metrics:
        do_better = den_better = tie = 0
        for r in rows:
            dv, ov = float(r[dcol]), float(r[ocol])
            if dv == ov:
                tie += 1
            elif (dv > ov) if lower_better else (dv < ov):
                do_better += 1
            else:
                den_better += 1
        out.append([name, do_better, den_better, tie, len(rows)])
    write_csv(os.path.join(TAB_OUT, "table_matched_box_count_metric_summary.csv"), header, out)

    tex = ["% Auto-generated from data/matched_box_count_pairs.csv by "
           "scripts/make_matched_box_count_tables.py",
           r"\begin{table}[t]", r"  \centering",
           r"  \caption{Direction of each metric across the 20 matched selected-box-count "
           r"settings. The comparison is metric-specific: the lower distance-only threshold "
           r"has the lower lost-ratio in every setting, while mAP@0.5 and native nuScenes "
           r"mAP each favour the density gate in a small number of settings.}",
           r"  \label{tab:matched-box-count-metric-summary}",
           r"  \begin{tabular}{lccc}", r"    \toprule",
           r"    Metric & Distance-only & Density & Total \\", r"    \midrule"]
    for name, do_b, den_b, tie, tot in out:
        tex.append("    %s & %d & %d & %d \\\\" % (name.replace("@", "@"), do_b, den_b, tot))
    tex += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    write_tex(os.path.join(TAB_OUT, "table_matched_box_count_metric_summary.tex"), tex)


if __name__ == "__main__":
    main()
