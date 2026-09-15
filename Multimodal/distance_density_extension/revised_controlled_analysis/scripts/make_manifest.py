"""Write docs/manifest.csv for the shipped sanitized data files.

Columns: public file (relative to this directory), SHA-256 of the shipped file,
number of data rows, and a short provenance category. No absolute paths, no
internal file names.

Run:  python scripts/make_manifest.py
"""
import csv
import hashlib
import os
from _common import DATA, DOC_OUT, ROOT, write_csv

CATEGORY = {
    "matched_box_count_pairs.csv":
        "controlled matched selected-box-count comparison, 20 settings",
    "controlled_conditions.csv":
        "controlled per-condition detection metrics, full-sensor baseline plus 40 pruning conditions",
    "controlled_matched_distance.csv":
        "controlled matched-distance comparison, full distance-only and p80/p90 gates at five thresholds",
    "point_budget_match_p90.csv":
        "p90 approximately matched deduplicated-point-budget construction and evaluation, two arms",
    "classwise_metrics.csv":
        "per-class detection metrics for every controlled condition",
    "classwise_selection_composition.csv":
        "per-class selection composition for every pruning condition",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def rows_of(path):
    with open(path, newline="") as fh:
        return max(0, sum(1 for _ in csv.reader(fh)) - 1)


def main():
    out = []
    for name in sorted(CATEGORY):
        p = os.path.join(DATA, name)
        out.append([os.path.join("data", name), sha256(p), rows_of(p), CATEGORY[name]])
    write_csv(os.path.join(DOC_OUT, "manifest.csv"),
              ["public_file", "sha256", "row_count", "source_category"], out)


if __name__ == "__main__":
    main()
