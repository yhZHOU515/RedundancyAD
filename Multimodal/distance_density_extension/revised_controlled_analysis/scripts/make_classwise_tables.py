"""Class-wise selection composition and class-wise lost-ratio tables (p90 gate).

Reads   data/classwise_selection_composition.csv   (per-class selection counts)
        data/classwise_metrics.csv                 (per-class detection metrics)
Writes  outputs/tables/table_classwise_selection_p90.{csv,tex}
        outputs/tables/table_classwise_lost_ratio_p90.{csv,tex}

The selection table shows which classes the two rules draw their selected boxes
from at a matched selected-box count; the lost-ratio table shows the per-class
detection cost of those selections. The p90 gate is shown as the representative
setting; the shipped CSVs cover all four percentiles.

Run:  python scripts/make_classwise_tables.py
"""
import os
from _common import DISTANCES, TAB_OUT, fmt_d, load, write_csv, write_tex

CLASSES = ["Car", "Pedestrian", "Cyclist"]
RULES = [("density", "density"), ("distance_only_matched", "distance-only")]


def main():
    sel = load("classwise_selection_composition.csv")
    met = load("classwise_metrics.csv")
    s_by = {(float(r["D"]), int(r["percentile"]), r["rule"], r["class_name"]): r for r in sel}
    m_by = {(r["condition_id"], r["class_name"]): r for r in met}

    # ---- selection composition -----------------------------------------
    header = ["T_dist_m", "class", "rule", "eligible_in_distance_gate", "selected",
              "selection_rate_within_class", "class_share_of_selection"]
    out = []
    for d in DISTANCES:
        for c in CLASSES:
            for rule, _ in RULES:
                r = s_by[(d, 90, rule, c)]
                out.append([fmt_d(d), c, rule, r["n_eligible_in_D_gate"], r["n_selected"],
                            "%.4f" % float(r["selection_rate_within_class"]),
                            "%.4f" % float(r["class_share_of_selection"])])
    write_csv(os.path.join(TAB_OUT, "table_classwise_selection_p90.csv"), header, out)

    tex = ["% Auto-generated from data/classwise_selection_composition.csv by "
           "scripts/make_classwise_tables.py",
           r"\begin{table}[t]", r"  \centering",
           r"  \caption{Class composition of the selected boxes at a matched selected-box "
           r"count, p90 gate. ``Share'' is the fraction of all selected boxes contributed "
           r"by the class; ``rate'' is the fraction of that class's distance-gated eligible "
           r"boxes that were selected.}",
           r"  \label{tab:classwise-selection-p90}",
           r"  \begin{tabular}{llcccc}", r"    \toprule",
           r"    & & \multicolumn{2}{c}{Density} & \multicolumn{2}{c}{Distance-only} \\",
           r"    \cmidrule(lr){3-4}\cmidrule(lr){5-6}",
           r"    $T_{\mathrm{dist}}$ & Class & Rate & Share & Rate & Share \\",
           r"    \midrule"]
    for i, d in enumerate(DISTANCES):
        if i:
            tex.append(r"    \midrule")
        for j, c in enumerate(CLASSES):
            den = s_by[(d, 90, "density", c)]
            dis = s_by[(d, 90, "distance_only_matched", c)]
            label = (r"    \multirow{3}{*}{%s\,m}" % fmt_d(d)) if j == 0 else "    "
            tex.append("%s & %s & %.3f & %.3f & %.3f & %.3f \\\\" % (
                label, c, float(den["selection_rate_within_class"]),
                float(den["class_share_of_selection"]),
                float(dis["selection_rate_within_class"]),
                float(dis["class_share_of_selection"])))
    tex += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    write_tex(os.path.join(TAB_OUT, "table_classwise_selection_p90.tex"), tex)

    # ---- class-wise lost-ratio -----------------------------------------
    header = ["T_dist_m", "class", "density_lost_ratio", "distance_only_lost_ratio",
              "delta_lost_ratio_density_minus_distance_only", "small_denominator_flag"]
    out = []
    for d in DISTANCES:
        dtag = fmt_d(d)
        for c in CLASSES:
            den = m_by[(f"D{dtag}_p90_density", c)]
            dis = m_by[(f"D{dtag}_p90_distance_only_matched", c)]
            dv, ov = float(den["lost_ratio"]), float(dis["lost_ratio"])
            flag = "True" if (den["small_denominator_flag"] == "True"
                              or dis["small_denominator_flag"] == "True") else "False"
            out.append([dtag, c, "%.5f" % dv, "%.5f" % ov, "%+.5f" % (dv - ov), flag])
    write_csv(os.path.join(TAB_OUT, "table_classwise_lost_ratio_p90.csv"), header, out)

    tex = ["% Auto-generated from data/classwise_metrics.csv by "
           "scripts/make_classwise_tables.py",
           r"\begin{table}[t]", r"  \centering",
           r"  \caption{Class-wise lost-ratio at a matched selected-box count, p90 gate. "
           r"$\Delta$ is density minus distance-only; positive values mean the density gate "
           r"loses more baseline true positives of that class.}",
           r"  \label{tab:classwise-lost-ratio-p90}",
           r"  \begin{tabular}{llccc}", r"    \toprule",
           r"    $T_{\mathrm{dist}}$ & Class & Density & Distance-only & $\Delta$ \\",
           r"    \midrule"]
    for i, d in enumerate(DISTANCES):
        if i:
            tex.append(r"    \midrule")
        for j, c in enumerate(CLASSES):
            row = out[i * len(CLASSES) + j]
            label = (r"    \multirow{3}{*}{%s\,m}" % fmt_d(d)) if j == 0 else "    "
            tex.append("%s & %s & %s & %s & $%s$ \\\\" % (label, c, row[2], row[3], row[4]))
    tex += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    write_tex(os.path.join(TAB_OUT, "table_classwise_lost_ratio_p90.tex"), tex)


if __name__ == "__main__":
    main()
