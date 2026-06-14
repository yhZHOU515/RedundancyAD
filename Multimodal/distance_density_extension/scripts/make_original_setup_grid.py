"""Original-setup 8x5 density-gate grid — verifier / compact summary.

This is a VERIFIER, not a figure/table generator: the original-setup component
ships only its result grid (no manuscript figure/table of its own). The script
reads the shipped 8x5 grid and prints a compact per-distance summary plus a
monotonicity sanity check, so a reader can confirm the grid's structure and
the headline trend without any raw data.

  reads   data/results/original_setup_grid_8x5.csv
  prints  per-T_dist lost-ratio by density gate (p00..p90) + monotonicity check

Setup: YOLO-LiDAR-Fusion (erosion=25, depth=20, pca=False, yolov8m-seg.pt) on
nuScenes-in-KITTI, 81-sample val split, DontCare-excluded pool
(Car/Pedestrian/Cyclist). p00 = distance-only; density percentiles are
recomputed within each distance-gated subset. Lost-ratio is at IoU 0.5 (Eq. 11).

Run from the package root:  python scripts/make_original_setup_grid.py
"""
import os
import csv

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GRID = os.path.join(ROOT, "data", "results", "original_setup_grid_8x5.csv")

T_DIST = [2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0]
MODES = [("dist_only", "p00"), ("p50", "p50"), ("p70", "p70"),
         ("p80", "p80"), ("p90", "p90")]


def load():
    rows = list(csv.DictReader(open(GRID)))
    return {(float(r["T_dist"]), r["mode"]): r for r in rows}


def main():
    idx = load()
    print(f"Original-setup density-gate grid: {len(idx)} cells "
          f"({len(T_DIST)} distance x {len(MODES)} density)\n")
    header = "T_dist(m) | " + " | ".join(f"{lbl:>7}" for _, lbl in MODES)
    print(header)
    print("-" * len(header))
    all_mono = True
    for T in T_DIST:
        seq = [float(idx[(T, m)]["lost50_overall"]) for m, _ in MODES]
        mono = all(a + 1e-9 >= b for a, b in zip(seq, seq[1:]))
        all_mono = all_mono and mono
        cells = " | ".join(f"{v:7.4f}" for v in seq)
        print(f"{T:>8g}  | {cells}   {'mono' if mono else 'NON-MONO'}")
    print("\nlost-ratio @ IoU 0.5 (lower = better); p00 = distance-only.")
    print(f"Monotone non-increasing in gate strictness (p00>=p50>=...>=p90) "
          f"at every T_dist: {all_mono}")
    print("\nReading: the conjunctive distance+density rule is more selective "
          "and controlled than distance-only — stricter density gates retain "
          "denser, better-supported objects and do not increase lost-ratio.")


if __name__ == "__main__":
    main()
