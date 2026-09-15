# Metric definitions

Every metric below is reported verbatim in the shipped CSVs. Nothing in this
directory recomputes a metric from raw data.

## Selection rules

For a LiDAR detection box `b`, `d(b)` is the ego-centric distance of the box
centroid and `rho(b) = n(b) / A_2D(b)` is the camera–LiDAR support density, where
`n(b)` is the number of LiDAR returns inside the 3D box and `A_2D(b)` is the pixel
area of the box's 2D image projection. Support density is therefore support **per
unit projected image area**, not a raw point count.

* **Distance-only.** `b` is selected when `d(b) <= T_dist`.
* **Distance–density (conjunctive).** `b` is selected when `d(b) <= T_dist` and
  `rho(b) >= T_rho`, where `T_rho` is a percentile of `rho` computed among the
  boxes that pass the distance gate. `p00` disables the density gate.
* **Box-count-matched distance-only.** The distance threshold is lowered below the
  outer `T_dist` until exactly the same number of boxes is selected as by the
  density gate at that setting. The resulting threshold is reported as
  `distance_only_effective_cutoff`.
* **Point-budget-matched distance-only.** The distance threshold is lowered until
  the cumulative number of removed **deduplicated** keyframe LiDAR points is
  closest to the number removed by the p90 gate. The selected-box count is an
  outcome of this construction, not a target.

Selected boxes are the pruning candidates; what the detector actually loses is the
**deduplicated union** of LiDAR points inside them, since overlapping boxes share
points. Selected-box count and removed-point count are therefore different budgets.

## Detection metrics

* **Lost-ratio** (primary). The fraction of the full-sensor baseline's true-positive
  detections that are no longer detected after pruning, matched at 3D IoU >= 0.5
  over the three candidate classes (Car, Pedestrian, Cyclist). Lower is better. It
  is always computed against the baseline run of the same fixed inference setup.
* **mAP@0.5.** Mean average precision at 3D IoU >= 0.5 over the same three classes.
* **Native nuScenes mAP.** The standard nuScenes detection protocol: mean AP over
  its ten classes, each averaged over the center-distance matching thresholds
  {0.5, 1, 2, 4} m. This is a **different class set and a different matching rule**
  from mAP@0.5, so the two are not directly comparable and neither contains the
  other.

## Denominators

* Eligible object pool: **74,464** Car, Pedestrian and Cyclist objects.
* Removed-point percentages: **34,548,128** deduplicated LiDAR points across the
  995 holdout keyframes.
* Lost-ratio denominator: the baseline true-positive count of the shared
  full-sensor baseline, **10,215** detections.

## Reading the comparisons

Differences are reported as **density minus distance-only**. For lost-ratio a
negative difference favours the density gate; for mAP@0.5 and native nuScenes mAP a
positive difference favours it. The direction is not the same for every metric, so
summaries must name the metric they refer to.
