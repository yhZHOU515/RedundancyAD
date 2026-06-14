# Reproduction notes

This package regenerates the journal-extension distance–density figures/tables
from included **aggregate/summary** inputs. It is not meant to reproduce the
entire conference paper, nor to re-run model inference end-to-end.

## What runs from this package

With the core Python stack (`pip install -r ../requirements.txt`) and no other
inputs:

- `scripts/make_diagnostic_figure_and_table.py` → diagnostic figure + table from
  `data/results/diagnostic_aggregate_stats.csv`.
- `scripts/make_holdout_figure_and_table.py` → holdout figure + table from
  `data/results/holdout_grid_5x5.csv`.
- `scripts/make_original_setup_grid.py` → console verifier/summary over
  `data/results/original_setup_grid_8x5.csv`.

The `data/results/` CSVs are **aggregate/summary verification files**: the
figure/table values are read verbatim from them. No reported number is
recomputed from raw data or changed by these scripts.

## `pipeline/` — provenance scripts (reference only)

`pipeline/` holds the upstream scripts that produced the `data/results/` CSVs,
grouped into `diagnostic/`, `bevfusion_holdout/`, and `original_setup/`. Each
carries a provenance banner naming its source in the full working tree. They are
**reference only** — not runnable from this package, because they need the
excluded raw data, checkpoints, framework source trees, and shared libraries
(`lib.py` / `lib3.py` / `holdout_lib.py` / `calibration.py`). They document the
exact methodology; the `scripts/` are what regenerate the figures/tables from the
included aggregate CSVs.

## Required external data / checkpoints (NOT included)

Re-running the *upstream* inference that produced the aggregate CSVs requires the
following, which are intentionally excluded (raw data and weights are large and
not redistributable here):

- nuScenes `v1.0-trainval` (+ mini) raw data and `LIDAR_TOP` sweeps.
- BEVFusion LiDAR–camera checkpoint + MMDetection3D (`projects/BEVFusion`).
- YOLO-LiDAR-Fusion code + `yolov8m-seg.pt` (original-setup component).
- Large intermediates: `*_infos_val.pkl`, per-object feature CSVs, per-cell
  prediction dumps.

These are not needed to regenerate the figures/tables above — only to recompute
the aggregate CSVs from scratch.

## Notes

- `holdout_grid_5x5.csv` contains the 25 density-gate cells (5 distance × 5
  density) plus one clearly-labeled `baseline` reference row (full-sensor, no
  removal). The baseline values are the verified full-sensor metrics; they make
  the holdout figure/table self-contained from this one file.
- The regenerated holdout outputs reflect the 5×5 grid distances
  {10, 15, 20, 22.5, 30} m. An exploratory 12.5 m distance-only point that
  appeared in an earlier detailed dump is not part of the 5×5 grid and is not
  included; no reported number changed.
- Splits: `part1_94scenes.txt` is the 94-scene Part-1 partition;
  `holdout_25scenes.txt` is `val_p01` minus the two mini_val scenes
  (`minival_scenes.txt` = scene-0103, scene-0916).
