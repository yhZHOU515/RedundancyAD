"""Shared helpers for the revised controlled analysis scripts.

Path resolution is relative to this file, so every script runs from anywhere:

    python scripts/make_matched_box_count_tables.py
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
TAB_OUT = os.path.join(ROOT, "outputs", "tables")
FIG_OUT = os.path.join(ROOT, "outputs", "figures")
DOC_OUT = os.path.join(ROOT, "docs")

DISTANCES = [10.0, 15.0, 20.0, 22.5, 30.0]
PERCENTILES = [50, 70, 80, 90]


def load(name):
    """Read a shipped CSV into a list of dicts (values kept as text)."""
    with open(os.path.join(DATA, name), newline="") as fh:
        return list(csv.DictReader(fh))


def fmt_d(d):
    """Format a distance threshold the way the manuscript prints it."""
    d = float(d)
    return ("%g" % d)


def write_csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    print("wrote", os.path.relpath(path, ROOT))


def write_tex(path, lines):
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", os.path.relpath(path, ROOT))
