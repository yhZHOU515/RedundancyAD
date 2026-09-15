# Provenance

## How these files were produced

The shipped CSVs are **sanitized aggregate derivatives** of the controlled
analysis outputs, drawn from three controlled experiments that share one saved
full-sensor baseline: the matched-distance runs, the matched selected-box-count
grid, and the approximately matched point-budget runs. Values were carried across
as text, so every number here is identical to the analysis output it came from.
The only exception is the two percentage columns in
`controlled_matched_distance.csv`, which are exact divisions of shipped integer
counts, introduced because the source files expressed these quantities in
different units.

Conditions that appear in more than one source file were checked against each
other: the shared full-sensor baseline and the p90 gate rows are identical across
all three experiments, and `scripts/verify_public_artifacts.py` re-checks the
overlapping rows on every run.

The sanitization applied to each file was:

* internal condition identifiers were renamed to the public scheme documented in
  [column_definitions.md](column_definitions.md), which removes references to
  internal execution infrastructure;
* the `pred_json` column, which held absolute filesystem paths to per-condition
  prediction dumps, was **removed entirely** from the per-condition file rather
  than shortened;
* an internal wall-clock timing column was removed, being run telemetry rather
  than a result;
* free-text provenance labels were rewritten to drop internal batch and machine
  names, keeping only whether a condition was newly run for this extension or
  reused unchanged;
* a cross-reference naming an internal file was repointed at the shipped public
  file.

Raw predictions, dataset files, checkpoints, internal experiment directories,
internal reports, run logs and abandoned attempts are not included.

## Integrity

[manifest.csv](manifest.csv) lists each shipped data file with its SHA-256, its
row count and a short provenance category. `scripts/verify_public_artifacts.py`
re-checks those hashes and row counts, re-derives every difference column from the
two level columns it summarises, reproduces the metric-direction counts quoted in
the README, and confirms that no shipped file contains an absolute path.

The per-condition file additionally carries `removed_id_hash` and
`point_mask_hash`, short hashes of the selected-object id set and of the per-frame
removal masks, plus `removed_id_hash_matches_frozen_design`, which records whether
the executed selection matched the design fixed before any detection metric was
computed.

## Relationship to the parent package

The parent package [`../../`](../../) is unchanged by this addition. Its
`data/results/holdout_grid_5x5.csv` and the tables and figures generated from it
remain the **submission-era** matched-distance results and are still the inputs for
the figures and tables in that directory.

The files in this directory come from the later controlled analysis, in which the
full-sensor baseline and every pruning condition were evaluated under one fixed
inference setup. The submission-era and revised controlled results are retained as
separate evaluation families and should not be combined. The revised manuscript uses
the common-baseline controlled values distributed in this directory.

## Ordering of the analysis

The selection design for every condition, including the matched cutoffs and the
point budgets, was fixed and written down before any detection metric was computed
for those conditions, which is what `removed_id_hash_matches_frozen_design`
records. The reviewer-requested matched selected-box-count comparison was added
afterwards and is reported here in full, including the settings where it does not
favour the density gate.
