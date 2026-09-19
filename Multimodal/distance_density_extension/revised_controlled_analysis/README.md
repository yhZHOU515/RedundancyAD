# Revised controlled analysis (camera–LiDAR distance–density)

This directory adds the reproducibility materials for the **revised** camera–LiDAR
distance–density analysis. The submission-era aggregate files, tables and figures
in the parent package [`../`](../) are retained separately for provenance.

The supplied figures and tables can be regenerated from the sanitized aggregate
CSVs in [data/](data/). Raw predictions, dataset files, model checkpoints and full
inference artifacts are **not** redistributed.

## Four families of results, and how they differ

The distance–density rule selects a box when `d(b) <= T_dist` **and**
`rho(b) >= T_rho`, where `T_rho` is a percentile of the support density computed
**within** the distance-gated eligible set. A p-th percentile gate therefore keeps
roughly the top `(100 - p)%` of that set, so at the same `T_dist` the density gate
selects far fewer boxes than the distance-only rule. Comparisons must state what
was held fixed:

| Family | What is held fixed | What differs | Where it lives |
|---|---|---|---|
| **1. Original / submission-era matched distance** | outer `T_dist` | selected-box count, removed points | parent package: [`../data/results/holdout_grid_5x5.csv`](../data/results/holdout_grid_5x5.csv) and the tables/figures generated from it |
| **2. Revised controlled matched distance** | outer `T_dist`, one fixed inference setup and one shared full-sensor baseline | selected-box count, removed points | [data/controlled_matched_distance.csv](data/controlled_matched_distance.csv): full distance-only pruning and the p80 and p90 gates at five thresholds |
| **3. Matched selected-box count** (the reviewer-requested comparison) | number of selected boxes, per setting | which boxes, removed points, effective cutoff | [data/matched_box_count_pairs.csv](data/matched_box_count_pairs.csv), all 20 settings |
| **4. Approximately matched point budget** (supporting) | number of removed deduplicated keyframe LiDAR points | selected-box count, which boxes | [data/point_budget_match_p90.csv](data/point_budget_match_p90.csv), p90 gate |

Families 1 and 2 answer "at the same distance threshold, what does adding the
density gate do?". Families 3 and 4 compare the rules at matched selected-box
counts and approximately matched deduplicated-point counts, respectively.

## What the shipped numbers show

Read metric by metric. The comparison does not point the same way for every metric,
and none of these files supports a claim that support density is universally
superior.

* **At matched distance thresholds**, density gating selects far fewer boxes than
  distance-only pruning and has a lower lost-ratio (families 1 and 2).
* **At a matched selected-box count**, the lower distance-only threshold has the
  lower lost-ratio in **all 20 of 20** settings
  ([table_matched_box_count_lost_ratio](outputs/tables/table_matched_box_count_lost_ratio.csv)).
* On **native nuScenes mAP**, the lower distance-only threshold is higher in
  **19 of 20** settings; on **mAP@0.5** it is higher in **17 of 20**. The density
  gate is higher in the remaining settings
  ([table_matched_box_count_metric_summary](outputs/tables/table_matched_box_count_metric_summary.csv)).
* **At an approximately matched point budget** (p90 gate, every setting matched to
  within 0.51% of the target point count), the two rules need very different
  numbers of boxes to remove approximately the same number of points. The re-cut
  distance-only comparator has the lower lost-ratio at all five thresholds
  ([table_point_budget_match_p90](outputs/tables/table_point_budget_match_p90.csv)).
  Selected-box count and removed-point volume are therefore different budgets and
  are not interchangeable.
* **Per class**, the direction is not uniform either: at a matched selected-box
  count the density gate loses more Car and Pedestrian true positives, while the
  Cyclist column moves the other way in several settings
  ([table_classwise_lost_ratio_p90](outputs/tables/table_classwise_lost_ratio_p90.csv)).

Support density characterizes **LiDAR support per projected image area** and
provides an additional criterion for restricting the candidate set within a
chosen distance range. On this holdout with the fixed pretrained model, the
matched selected-box-count comparisons do not establish better preservation of
baseline true positives through density gating.

## Quick start

Run these commands from this directory
(`Multimodal/distance_density_extension/revised_controlled_analysis/`):

```bash
python -m pip install -r ../requirements.txt
python scripts/make_controlled_matched_distance.py
python scripts/make_matched_box_count_tables.py
python scripts/make_matched_box_count_figures.py
python scripts/make_point_budget_table.py
python scripts/make_classwise_tables.py
python scripts/verify_public_artifacts.py
```

The figure and table scripts read the CSVs in [data/](data/) and write into
[outputs/](outputs/). `make_manifest.py` records the data-file hashes in
[docs/manifest.csv](docs/manifest.csv); `verify_public_artifacts.py` checks those
hashes and the reported numerical relationships.

## Results map

The map below identifies the detailed exports by LaTeX label or output name.
Their relationship to the final revised manuscript is:

* **Figure 8:** the controlled matched-distance figure uses
  [data/controlled_matched_distance.csv](data/controlled_matched_distance.csv).
* **Table VII:** the representative rows at 10, 20, and 30 m combine the full
  distance-only and p90 conditions from that file with the p90 matched
  selected-box-count comparators in
  [data/matched_box_count_pairs.csv](data/matched_box_count_pairs.csv).
* **Table VIII:** the class-wise breakdown uses the 30 m p90 rows in
  [data/classwise_selection_composition.csv](data/classwise_selection_composition.csv)
  and [data/classwise_metrics.csv](data/classwise_metrics.csv).

The supplied table scripts produce the detailed exports listed below; they do
not assemble the exact combined layouts of manuscript Tables VII and VIII.

| Label | Output | Script | Input |
|---|---|---|---|
| `tab:controlled-matched-distance-holdout` | [outputs/tables/table_controlled_matched_distance.tex](outputs/tables/table_controlled_matched_distance.tex) | `make_controlled_matched_distance.py` | [data/controlled_matched_distance.csv](data/controlled_matched_distance.csv) |
| controlled matched-distance figure | [outputs/figures/fig_controlled_matched_distance.pdf](outputs/figures/fig_controlled_matched_distance.pdf) | `make_controlled_matched_distance.py` | [data/controlled_matched_distance.csv](data/controlled_matched_distance.csv) |
| `tab:matched-box-count-holdout` | [outputs/tables/table_matched_box_count_lost_ratio.tex](outputs/tables/table_matched_box_count_lost_ratio.tex) | `make_matched_box_count_tables.py` | [data/matched_box_count_pairs.csv](data/matched_box_count_pairs.csv) |
| `tab:matched-box-count-metric-summary` | [outputs/tables/table_matched_box_count_metric_summary.tex](outputs/tables/table_matched_box_count_metric_summary.tex) | `make_matched_box_count_tables.py` | [data/matched_box_count_pairs.csv](data/matched_box_count_pairs.csv) |
| `tab:point-budget-match-p90` | [outputs/tables/table_point_budget_match_p90.tex](outputs/tables/table_point_budget_match_p90.tex) | `make_point_budget_table.py` | [data/point_budget_match_p90.csv](data/point_budget_match_p90.csv) |
| `tab:classwise-selection-p90` | [outputs/tables/table_classwise_selection_p90.tex](outputs/tables/table_classwise_selection_p90.tex) | `make_classwise_tables.py` | [data/classwise_selection_composition.csv](data/classwise_selection_composition.csv) |
| `tab:classwise-lost-ratio-p90` | [outputs/tables/table_classwise_lost_ratio_p90.tex](outputs/tables/table_classwise_lost_ratio_p90.tex) | `make_classwise_tables.py` | [data/classwise_metrics.csv](data/classwise_metrics.csv) |
| matched box-count lost-ratio figure | [outputs/figures/fig_matched_box_count_lost_ratio.pdf](outputs/figures/fig_matched_box_count_lost_ratio.pdf) | `make_matched_box_count_figures.py` | [data/matched_box_count_pairs.csv](data/matched_box_count_pairs.csv) |
| per-setting metric differences figure | [outputs/figures/fig_matched_box_count_metric_deltas.pdf](outputs/figures/fig_matched_box_count_metric_deltas.pdf) | `make_matched_box_count_figures.py` | [data/matched_box_count_pairs.csv](data/matched_box_count_pairs.csv) |

Each table script writes the `.csv` next to the `.tex`; the figure script writes
`.png` next to the `.pdf`.

## Layout

```text
README.md
data/      controlled_matched_distance.csv · matched_box_count_pairs.csv
           controlled_conditions.csv · point_budget_match_p90.csv
           classwise_metrics.csv · classwise_selection_composition.csv
outputs/tables/   six table pairs (.csv + .tex)
outputs/figures/  three figures (.pdf + .png)
scripts/   make_controlled_matched_distance.py · make_matched_box_count_tables.py
           make_matched_box_count_figures.py · make_point_budget_table.py
           make_classwise_tables.py · make_manifest.py
           verify_public_artifacts.py · _common.py
docs/      metric_definitions.md · column_definitions.md · inference_setup.md
           provenance.md · manifest.csv
```

## Scope limits

This directory covers the camera–LiDAR analysis only. See the [repository scope](../../../readme.md#reproducibility-materials) for the available camera–camera materials.

* The comparisons here are for **one 25-scene / 995-keyframe nuScenes holdout** and
  **one fixed pretrained BEVFusion checkpoint**, evaluated in inference-only mode.
  They are not evidence about other datasets, detectors, or training regimes.
* Counts such as "20 of 20" are counts of settings, **not statistical tests**. No
  confidence intervals or significance claims are attached to any value here.
* The matched selected-box-count family covers all four percentiles. The point
  budget family covers the **p90 gate only**.
* The matched-distance file covers the conditions drawn in the controlled
  matched-distance figure: full distance-only pruning and the p80 and p90 gates.
  The p50 and p70 gates appear in the matched selected-box-count files.
* Pruning removes LiDAR returns from the **current keyframe scan only**; the
  preceding sweeps aggregated by the BEVFusion input pipeline are unchanged. See
  [docs/inference_setup.md](docs/inference_setup.md).
