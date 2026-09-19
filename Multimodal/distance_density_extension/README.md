# Camera–LiDAR Distance–Density (Journal Extension)

The camera–LiDAR **distance–density** experiments added in the RedundancyAD
journal extension. This directory covers only that analysis; it provides:

* the shipped aggregate/summary inputs under [data/](data/);
* lightweight scripts that regenerate the supplied figures/tables from those inputs;
* reference provenance scripts under [pipeline/](pipeline/) documenting how the
  aggregate inputs were produced;
* the original diagnostic and submission-era holdout figures/tables under [outputs/](outputs/);
* the aggregate reproducibility materials for the **revised** manuscript under
  [revised_controlled_analysis/](revised_controlled_analysis/).

The original diagnostics and **submission-era** holdout results remain under
`data/`, `outputs/` and `scripts/`. For the revised BEVFusion evaluation, use
`revised_controlled_analysis/`; its quick start is shown first below.

Raw datasets, pretrained checkpoints, and full inference artifacts are **not
redistributed** here (dataset licensing and storage constraints).

The distance-**only** camera–LiDAR baseline that this module extends is kept
separately in the sibling directory
[`../nuScenes/`](../nuScenes/).

## Method

For each LiDAR detection box `b`, we use two operational signals for selecting
removal candidates: the ego-centric distance of the box centroid `d(b)` and
LiDAR support density `rho(b)`. These signals do not establish that the selected
information is strictly redundant.

**Distance gate (baseline).** The original distance-only rule (in
[`../nuScenes/`](../nuScenes/)) treats a box as a pruning candidate when its
3D centroid lies **within** a distance threshold — i.e., close-range boxes are
removed:

> B_pruned = { b ∈ B_LiDAR | d(b) ≤ T_dist }

**Support density.** Distance alone is an incomplete signal: within the near
range, many objects still receive sparse LiDAR returns (the diagnostic analysis
shows 45.3% of near-range object–camera rows have fewer than 10 in-box points).
We therefore add a support-density measure,

> rho(b) = n(b) / A_2D(b)

where `n(b)` is the number of LiDAR returns inside the 3D box and `A_2D(b)` is
the pixel area of the box's 2D projection in the image plane. Higher `rho(b)`
means more LiDAR returns per unit projected image area; it does not measure
semantic agreement between camera and LiDAR observations.

**Conjunctive distance–density rule (this module).** A box is a pruning
candidate only if it passes **both** gates:

> B_cand = { b ∈ B_LiDAR | d(b) ≤ T_dist  ∧  rho(b) ≥ T_rho }

`T_rho` is a percentile threshold computed **among the distance-gated eligible
boxes**. Setting the density gate to `p00` disables it and
recovers the distance-only baseline; higher percentiles (p80, p90) restrict
pruning to near-range boxes with higher LiDAR support density.

This module evaluates the rule on a pretrained BEVFusion model over a 25-scene
nuScenes holdout. In the **submission-era** results kept here, at `T_dist = 30 m`
distance-only pruning (`p00`) removes 57.9% of the eligible pool (lost-ratio
0.104) while the `p90` gate removes 5.8% (lost-ratio 0.050): at the same outer
threshold the gate restricts the candidate set, so it selects roughly 10× fewer
boxes and has the lower lost-ratio. All threshold combinations are in
[data/results/holdout_grid_5x5.csv](data/results/holdout_grid_5x5.csv).

The revised manuscript re-evaluates these conditions against one common saved
full-sensor baseline under a fixed inference seed, and adds comparisons at
matched selected-box counts and approximately matched deduplicated-point counts.
Those results are in
[revised_controlled_analysis/](revised_controlled_analysis/), which also states
the outcome metric by metric: at a matched selected-box count the re-cut
distance-only rule has the lower lost-ratio in all 20 comparisons, the higher
native nuScenes mAP in 19 of 20 and the higher mAP@0.5 in 17 of 20, and the
supporting p90 approximately matched deduplicated-point comparison preserves the
same lost-ratio ordering, favouring the matched distance-only comparator.
Selected-box count and removed-point volume are different budgets.
Support density remains an object-level measure of LiDAR support per projected
image area and an additional candidate-restriction criterion; these results do not
establish universally better detection preservation. The full-sensor model remains
the reference point for detection performance.

## Quick start

**Revised controlled BEVFusion analysis.** From the repository root:

```bash
cd Multimodal/distance_density_extension/revised_controlled_analysis
python -m pip install -r ../requirements.txt
python scripts/verify_public_artifacts.py
python scripts/make_controlled_matched_distance.py
python scripts/make_matched_box_count_tables.py
python scripts/make_matched_box_count_figures.py
python scripts/make_point_budget_table.py
python scripts/make_classwise_tables.py
```

These commands verify the supplied data and regenerate the revised controlled
outputs from aggregate CSV files. They do not rerun model inference. See the
[revised results map](revised_controlled_analysis/README.md#results-map) for the
relationship between the detailed exports and the final manuscript tables.

### Original diagnostics and submission-era outputs

The following commands run from this directory
(`Multimodal/distance_density_extension/`):

```bash
python -m pip install -r requirements.txt
python scripts/make_diagnostic_figure_and_table.py
python scripts/make_holdout_figure_and_table.py
python scripts/make_original_setup_grid.py
```

`make_holdout_figure_and_table.py` regenerates the **submission-era** BEVFusion
holdout outputs. The revised controlled holdout outputs come from the separate
directory above.

**Requirements files.** `requirements.txt` contains the lightweight dependencies
(`numpy`, `pandas`, `matplotlib`) for aggregate-output regeneration.
`requirements-pipeline.txt` is for the reference code under `pipeline/`; it is
not needed for the quick start.

## Results map

**Original diagnostics and submission-era outputs.**

The holdout figure and table in this map are historical outputs. For the revised
holdout results, use the [controlled-analysis results map](revised_controlled_analysis/README.md#results-map).

| Label                           | Output                                                                                                   | Script                                                  | Input                                                                                      |
| ------------------------------- | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `fig:diagnostic-cl-redundancy`  | [outputs/figures/fig_diagnostic_cl_redundancy.pdf](outputs/figures/fig_diagnostic_cl_redundancy.pdf)     | `python scripts/make_diagnostic_figure_and_table.py`    | [data/results/diagnostic_aggregate_stats.csv](data/results/diagnostic_aggregate_stats.csv) |
| `tab:cl-redundancy-diagnostic`  | [outputs/tables/table_cl_redundancy_main.tex](outputs/tables/table_cl_redundancy_main.tex)               | `python scripts/make_diagnostic_figure_and_table.py`    | [data/results/diagnostic_aggregate_stats.csv](data/results/diagnostic_aggregate_stats.csv) |
| `fig:holdout-matched-threshold` | [outputs/figures/fig_holdout_matched_threshold.pdf](outputs/figures/fig_holdout_matched_threshold.pdf)   | `python scripts/make_holdout_figure_and_table.py`       | [data/results/holdout_grid_5x5.csv](data/results/holdout_grid_5x5.csv)                     |
| `tab:matched-tdist-holdout`     | [outputs/tables/table_holdout_matched_threshold.tex](outputs/tables/table_holdout_matched_threshold.tex) | `python scripts/make_holdout_figure_and_table.py`       | [data/results/holdout_grid_5x5.csv](data/results/holdout_grid_5x5.csv)                     |
| Complete BEVFusion 5×5 grid     | [data/results/holdout_grid_5x5.csv](data/results/holdout_grid_5x5.csv)                                   | —                                                       | —                                                                                          |
| Original setup 8×5 grid         | [data/results/original_setup_grid_8x5.csv](data/results/original_setup_grid_8x5.csv)                     | `python scripts/make_original_setup_grid.py` (verifier) | [data/results/original_setup_grid_8x5.csv](data/results/original_setup_grid_8x5.csv)       |

Each figure script also writes the `.png` figure and the `.csv` table next to the
`.pdf`/`.tex` listed above. `make_original_setup_grid.py` is a **verifier**: it
prints a compact per-distance summary and a monotonicity sanity check over the
8×5 grid (no figure/table of its own is part of the manuscript).

## Layout

```text
README.md  requirements.txt  requirements-pipeline.txt  .gitignore
scripts/
  make_diagnostic_figure_and_table.py     # reads data/results/diagnostic_aggregate_stats.csv
  make_holdout_figure_and_table.py        # reads data/results/holdout_grid_5x5.csv
  make_original_setup_grid.py             # verifier over data/results/original_setup_grid_8x5.csv
  lost_ratio.py                           # lost-ratio metric, standalone reference
pipeline/                                 # provenance scripts (reference only — see below)
  diagnostic/         build_object_camera_table.py · aggregate_diagnostic_stats.py
  bevfusion_holdout/  build_holdout_split.py · build_eligible_pool.py · prune_lidar_returns.py
                      run_bevfusion_grid.py · compute_holdout_grid.py
  original_setup/     run_original_density_grid.py · compute_original_setup_grid.py
data/splits/    part1_94scenes.txt · holdout_25scenes.txt · minival_scenes.txt
data/results/   diagnostic_aggregate_stats.csv · holdout_grid_5x5.csv · original_setup_grid_8x5.csv
outputs/figures/  fig_diagnostic_cl_redundancy.{pdf,png} · fig_holdout_matched_threshold.{pdf,png}
outputs/tables/   table_cl_redundancy_main.{csv,tex} · table_holdout_matched_threshold.{csv,tex}
docs/reproduction_notes.md
revised_controlled_analysis/   # aggregate materials for the revised manuscript
```

The lost-ratio metric is defined standalone in
[scripts/lost_ratio.py](scripts/lost_ratio.py). The `data/results/` files are
**aggregate/summary verification files** — the figure/table numbers are read
verbatim from them; the scripts do not change any reported value. See
[docs/reproduction_notes.md](docs/reproduction_notes.md).

## `pipeline/` — provenance (reference only)

`pipeline/` documents **how the `data/results/` CSVs were produced**, organized by
component. These are reference scripts copied from the full working tree (with a
provenance banner at the top of each); they are **not runnable from this package
alone** — they require the excluded raw data (nuScenes), model checkpoints,
framework source trees (MMDetection3D / BEVFusion, YOLO-LiDAR-Fusion), and shared
geometry/eval libraries from the full tree. They are included so readers can
read the exact methodology behind each result.

| Component         | Pipeline (in order)                                                                                                                  | Produces                                      |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------- |
| diagnostic        | `build_object_camera_table.py` → `aggregate_diagnostic_stats.py`                                                                     | `data/results/diagnostic_aggregate_stats.csv` |
| bevfusion holdout | `build_holdout_split.py` → `build_eligible_pool.py` → `prune_lidar_returns.py` → `run_bevfusion_grid.py` → `compute_holdout_grid.py` | `data/results/holdout_grid_5x5.csv`           |
| original setup    | `run_original_density_grid.py` → `compute_original_setup_grid.py`                                                                    | `data/results/original_setup_grid_8x5.csv`    |
