# Inference setup

## Model and data

* A single pretrained BEVFusion LiDAR–camera checkpoint, run **inference only**.
  No retraining and no fine-tuning on pruned inputs.
* A held-out 25-scene nuScenes subset containing 995 keyframes. The scene list is
  shipped in the parent package at
  [`../../data/splits/holdout_25scenes.txt`](../../data/splits/holdout_25scenes.txt).
* Pruning candidates are Car, Pedestrian and Cyclist detection boxes.

## Common inference setup

All conditions in this directory, including the full-sensor baseline, were
evaluated under **one fixed inference setup with a fixed random seed**, so every
pruned condition is compared against the same baseline predictions. This matters
because lost-ratio is defined relative to the baseline's matched true positives:
pairing predictions from different runs changes the value even when the removed
object set is identical.

The shared baseline shipped in [../data/controlled_conditions.csv](../data/controlled_conditions.csv)
is the row `baseline_full_sensor`: mAP@0.5 0.4049039684182132, native nuScenes mAP
0.673120629207898, 10,215 baseline true positives, zero objects removed.

Repeat runs of one condition under this setup reproduced the removal set exactly
and the detection metrics to within a small tolerance, so the values here are
reproducible but not bit-exact across runs. No significance test is attached to
any difference.

## Scope of point removal

Pruning removes the LiDAR returns that fall inside the selected boxes **from the
current keyframe scan only**. The preceding sweeps that the BEVFusion input
pipeline aggregates are left unchanged. Removed-point counts in these files are
therefore keyframe points, and the effect of also removing an object's returns
from earlier sweeps has not been evaluated.

## What is not included

Raw nuScenes data, the pretrained checkpoint, per-frame prediction dumps,
per-object feature tables and other large intermediates are not redistributed,
for licensing and storage reasons. The shipped aggregate CSVs are sufficient to
regenerate every table and figure in this directory.
