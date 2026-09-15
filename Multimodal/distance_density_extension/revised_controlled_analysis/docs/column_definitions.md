# Column definitions

All five files are aggregate summaries. Values are carried over verbatim from the
analysis outputs; the scripts in `scripts/` never alter a reported number.

## Shared identifiers

* `condition_id` — public identifier of one evaluated condition, formed as
  `D<T_dist>_p<percentile>_<rule>`, for example `D22.5_p90_density` or
  `D30_p90_distance_only_matched`. The full-sensor reference row is
  `baseline_full_sensor`.
* `rule` — `density`, `distance_only_matched`, or `full_sensor_baseline`.
* `D` — outer distance threshold `T_dist` in metres.
* `percentile` — density percentile `T_rho` (50, 70, 80 or 90).

## data/controlled_matched_distance.csv (16 rows)

The controlled matched-distance comparison: one shared full-sensor baseline row,
full distance-only pruning at each of the five outer thresholds, and the p80 and
p90 density gates at the same thresholds. Every row was evaluated against the same
saved baseline under one fixed inference seed.

| Column | Meaning |
|---|---|
| `condition_id` | public condition identifier, `baseline_full_sensor` for the reference row |
| `method` | `full_sensor_baseline`, `distance_only`, or `distance_density` |
| `T_dist_m` | outer distance threshold in metres |
| `density_percentile` | density percentile, empty for distance-only and for the baseline |
| `eligible_pool_count` | 74,464, the Car, Pedestrian and Cyclist eligible pool |
| `distance_gated_pool_count` | boxes inside the distance gate at this threshold |
| `selected_box_count` | boxes selected for pruning |
| `selected_pct_of_eligible_pool` | selected boxes as a percentage of 74,464, **derived from the two counts** |
| `dedup_points_removed` | deduplicated keyframe LiDAR points removed |
| `dedup_points_removed_pct_of_holdout` | removed points as a percentage of 34,548,128, **derived from the two counts** |
| `lost_ratio`, `mAP50`, `native_mAP` | detection metrics for this condition |
| `baseline_TP_count` | baseline true positives used as the lost-ratio denominator |
| `provenance_category` | whether the condition was run for this extension or reused unchanged |

The two derived percentage columns are the only computed values in any shipped
file. Both are exact divisions of the shipped integer counts, and
`scripts/verify_public_artifacts.py` re-derives them on every run. Selected-box
percentage and removed-point percentage are different quantities and move
independently: a rule can select many more boxes while removing fewer points.

## data/matched_box_count_pairs.csv (20 rows)

One row per (`T_dist`, percentile) setting, pairing the density gate with the
distance-only rule matched to the same selected-box count.

| Column | Meaning |
|---|---|
| `matched_N` | selected-box count shared by both rules in this setting |
| `density_threshold_val` | numeric support-density threshold at this percentile |
| `distance_only_effective_cutoff` | lowered distance threshold (m) that selects `matched_N` boxes |
| `density_points_removed`, `do_points_removed` | deduplicated keyframe LiDAR points removed |
| `density_points_removed_pct`, `do_points_removed_pct` | the same as a percentage of 34,548,128 |
| `density_baseline_TP`, `do_baseline_TP` | baseline true positives used as the lost-ratio denominator |
| `density_lost_TP`, `do_lost_TP` | baseline true positives no longer detected |
| `density_lost_ratio`, `do_lost_ratio` | lost-ratio of each rule |
| `density_mAP50`, `do_mAP50` | mAP@0.5 of each rule |
| `density_native_mAP`, `do_native_mAP` | native nuScenes mAP of each rule |
| `delta_lost_ratio`, `delta_mAP50`, `delta_native_mAP` | density minus distance-only |
| `provenance` | whether the condition was run for this extension or reused unchanged |

## data/controlled_conditions.csv (41 rows)

One row per evaluated condition: the full-sensor baseline, 20 density conditions
and their 20 box-count-matched distance-only counterparts.

| Column | Meaning |
|---|---|
| `removed_object_count`, `removed_pct_of_74464` | selected boxes, absolute and as a share of the eligible pool |
| `baseline_TP_count`, `lost_TP_count`, `lost_ratio` | lost-ratio and its components |
| `mAP50`, `native_mAP` | detection metrics for this condition |
| `density_threshold_val`, `effective_cutoff` | the thresholds actually applied |
| `outer_pool_count`, `removed_pct_of_outer_pool` | size of the distance-gated pool and the share of it selected |
| `removed_id_hash` | short hash of the selected-object id set, for integrity checking |
| `removed_id_hash_matches_frozen_design` | whether the executed selection matched the pre-registered design |
| `dedup_points_removed`, `dedup_points_removed_pct_of_34548128` | removed deduplicated keyframe points |
| `point_mask_hash` | short hash of the per-frame removal masks |
| `points_per_removed_object` | mean removed points per selected box |
| `provenance` | whether the condition was run for this extension or reused unchanged |

## data/point_budget_match_p90.csv (10 rows)

Two rows per outer distance threshold: the p90 gate and the distance-only rule
re-cut to remove approximately the same number of deduplicated keyframe points.
Both arms were evaluated against the same saved baseline, so their detection
metrics are directly comparable.

| Column | Meaning |
|---|---|
| `T_dist_m` | outer distance threshold that defines the budget |
| `arm` | `distance_density_p90` or `distance_only_point_matched` |
| `distance_cutoff_m` | distance threshold actually applied: the outer threshold for the gate, the lowered cutoff for the comparator |
| `target_dedup_points` | the p90 gate's removed points, used as the budget for both arms |
| `achieved_dedup_points` | points this arm actually removed |
| `rel_point_mismatch` | relative difference from the budget (fraction, not percent) |
| `selected_box_count` | boxes selected; an outcome of the point matching, not a target |
| `lost_ratio`, `mAP50`, `native_mAP` | detection metrics for this arm |
| `baseline_condition_id`, `baseline_TP_count` | the shared baseline and its true-positive count |
| `provenance_category` | whether the arm was run for this extension or reused unchanged |

## data/classwise_metrics.csv (123 rows)

Three rows per condition, one per class.

| Column | Meaning |
|---|---|
| `n_pred`, `n_GT`, `n_TP` | predictions, ground-truth objects and true positives for the class |
| `AP50`, `recall` | per-class detection metrics at IoU >= 0.5 |
| `baseline_TP`, `n_lost`, `lost_ratio` | per-class lost-ratio and its components |
| `small_denominator_flag` | set when the class denominator is small enough that the ratio is unstable |

## data/classwise_selection_composition.csv (120 rows)

Three rows per pruning condition, one per class. Describes what was selected, not
what was detected.

| Column | Meaning |
|---|---|
| `n_eligible_in_D_gate`, `n_eligible_total_all_classes` | class and total eligible boxes inside the distance gate |
| `n_selected`, `n_selected_total_all_classes` | class and total selected boxes |
| `selection_rate_within_class` | selected divided by eligible, within the class |
| `class_share_of_selection` | the class's share of all selected boxes |
| `small_eligible_denominator_flag`, `zero_eligible_denominator_flag` | small or empty eligible denominators |
| `detection_metrics_available_in` | pointer to the file holding the matching detection metrics |
| `per_class_point_removal` | marked unavailable: point removal is tracked per frame, not per class |
