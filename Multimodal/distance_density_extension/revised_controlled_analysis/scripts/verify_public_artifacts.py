"""Self-check for the shipped sanitized data and the numbers quoted in the docs.

Checks, using only files inside this directory:
  1. every data file matches its SHA-256 and row count in docs/manifest.csv;
  2. each difference column equals the difference of the two level columns;
  3. the metric-direction counts quoted in the README are reproduced from data;
  4. the shipped condition ids are consistent across the three per-condition files;
  5. the point budget is matched to better than 1% in every setting;
  5. the controlled matched-distance percentages match the shipped counts, and the
     conditions it shares with the box-count file carry identical metrics;
  6. the point-matched distance-only comparator has the lower lost-ratio at every
     threshold, and both arms cite the same baseline;
  7. no shipped data file contains an absolute filesystem path.

Exits non-zero if any check fails.

Run:  python scripts/verify_public_artifacts.py
"""
import csv
import os
import sys
from make_manifest import rows_of, sha256
from _common import DATA, DOC_OUT, load

TOL = 1e-9
failures = []


def check(ok, label, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + label + (f"  [{detail}]" if detail else ""))
    if not ok:
        failures.append(label)


def main():
    print("1. manifest integrity")
    with open(os.path.join(DOC_OUT, "manifest.csv"), newline="") as fh:
        manifest = list(csv.DictReader(fh))
    for m in manifest:
        p = os.path.join(os.path.dirname(DATA), m["public_file"])
        check(os.path.exists(p), f"{m['public_file']} present")
        if os.path.exists(p):
            check(sha256(p) == m["sha256"], f"{m['public_file']} sha256")
            check(rows_of(p) == int(m["row_count"]),
                  f"{m['public_file']} row count", f"expected {m['row_count']}")

    print("2. difference columns equal the difference of levels")
    pairs = load("matched_box_count_pairs.csv")
    for a, b, d in [("density_lost_ratio", "do_lost_ratio", "delta_lost_ratio"),
                    ("density_mAP50", "do_mAP50", "delta_mAP50"),
                    ("density_native_mAP", "do_native_mAP", "delta_native_mAP")]:
        worst = max(abs((float(r[a]) - float(r[b])) - float(r[d])) for r in pairs)
        check(worst < 1e-6, f"{d} consistent", f"max abs error {worst:.2e}")

    print("3. metric-direction counts")
    expect = {"lost_ratio": (20, 0), "mAP50": (17, 3), "native_mAP": (19, 1)}
    for key, (exp_do, exp_den) in expect.items():
        if key == "lost_ratio":
            dcol, ocol, lower_better = "density_lost_ratio", "do_lost_ratio", True
        elif key == "mAP50":
            dcol, ocol, lower_better = "density_mAP50", "do_mAP50", False
        else:
            dcol, ocol, lower_better = "density_native_mAP", "do_native_mAP", False
        do_b = sum(1 for r in pairs if ((float(r[dcol]) > float(r[ocol])) if lower_better
                                        else (float(r[dcol]) < float(r[ocol]))))
        den_b = len(pairs) - do_b
        check((do_b, den_b) == (exp_do, exp_den), f"{key} direction counts",
              f"distance-only {do_b}, density {den_b}")

    print("4. condition ids consistent across files")
    cond = {r["condition_id"] for r in load("controlled_conditions.csv")}
    met = {r["condition_id"] for r in load("classwise_metrics.csv")}
    sel = {r["condition_id"] for r in load("classwise_selection_composition.csv")}
    check(cond == met, "conditions vs class-wise metrics", f"{len(cond)} vs {len(met)}")
    check(sel <= cond, "selection ids are a subset of conditions",
          f"{len(sel)} of {len(cond)}")
    check("baseline_full_sensor" in cond, "full-sensor baseline row present")

    print("5. controlled matched-distance file")
    cmd = load("controlled_matched_distance.csv")
    pruning = [r for r in cmd if r["method"] != "full_sensor_baseline"]
    check(len(cmd) == 16 and len(pruning) == 15, "16 rows: baseline plus 15 conditions",
          f"{len(cmd)} rows")
    worst_pct = 0.0
    for r in pruning:
        for count, denom, col in [(r["selected_box_count"], 74464,
                                   "selected_pct_of_eligible_pool"),
                                  (r["dedup_points_removed"], 34548128,
                                   "dedup_points_removed_pct_of_holdout")]:
            worst_pct = max(worst_pct, abs(int(count) / denom * 100.0 - float(r[col])))
    check(worst_pct < 1e-9, "percentage columns match the shipped counts",
          f"max abs error {worst_pct:.2e}")
    shared = {r["condition_id"]: r for r in load("controlled_conditions.csv")}
    agree = mismatch = 0
    for r in cmd:
        o = shared.get(r["condition_id"])
        if not o:
            continue
        same = all(abs(float(r[c]) - float(o[c])) < 1e-12
                   for c in ("lost_ratio", "mAP50", "native_mAP"))
        agree += same
        mismatch += (not same)
    check(mismatch == 0 and agree >= 6, "conditions shared with the box-count file agree",
          f"{agree} shared rows, {mismatch} disagreements")

    print("6. point-budget arms")
    pb = load("point_budget_match_p90.csv")
    check(len(pb) == 10, "10 rows: two arms at five thresholds", f"{len(pb)} rows")
    worst = max(abs(float(r["rel_point_mismatch"])) for r in pb) * 100.0
    check(worst < 1.0, "point budget mismatch under 1%", f"worst {worst:.3f}%")
    check(abs(worst - 0.506) < 0.001, "worst mismatch is 0.506%", f"{worst:.4f}%")
    thresholds = sorted({float(r["T_dist_m"]) for r in pb})
    lower = 0
    for d in thresholds:
        p90 = [r for r in pb if float(r["T_dist_m"]) == d
               and r["arm"] == "distance_density_p90"][0]
        dor = [r for r in pb if float(r["T_dist_m"]) == d
               and r["arm"] == "distance_only_point_matched"][0]
        lower += float(dor["lost_ratio"]) < float(p90["lost_ratio"])
    check(lower == len(thresholds), "point-matched distance-only has the lower lost-ratio",
          f"{lower} of {len(thresholds)} thresholds")
    check({r["baseline_condition_id"] for r in pb} == {"baseline_full_sensor"},
          "both arms cite the same baseline")

    print("7. no absolute paths in shipped data")
    for m in manifest:
        p = os.path.join(os.path.dirname(DATA), m["public_file"])
        with open(p, encoding="utf-8", errors="replace") as fh:
            txt = fh.read()
        check("/home/" not in txt and "/mnt/" not in txt and "C:\\" not in txt,
              f"{m['public_file']} free of absolute paths")

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s)")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
