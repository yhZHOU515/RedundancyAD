# ---------------------------------------------------------------------------
# Provenance: original_camera_lidar_density_gate/run_density_gate_original_setup.py
# Role: Run the 8x5 distance x density grid on the conference YOLO-LiDAR-Fusion setup -> data/results/original_setup_grid_8x5.csv.
#
# REFERENCE / PROVENANCE SCRIPT -- documents how the shipped data/results CSVs
# were produced. NOT runnable from this package alone: it requires the
# excluded raw data (nuScenes), model checkpoints, framework source trees
# (MMDetection3D / BEVFusion, YOLO-LiDAR-Fusion), and the shared geometry/eval
# libraries from the full working tree (lib.py / lib3.py / holdout_lib.py /
# calibration.py). Absolute paths appear as /path/to/... placeholders.
# ---------------------------------------------------------------------------

"""Step 1.7 — density-gate sensitivity sweep on the Phase 1 conference setup.

Parallels Step 3.12 (BEVFusion holdout) on the Phase 1 stack:
  Model:   YOLO-LiDAR-Fusion (erosion=25, depth=20, pca=False, yolov8m-seg.pt)
  Dataset: nuScenes-in-KITTI, 81-sample val split
  Pool:    DontCare-excluded {Car, Pedestrian, Cyclist}

Grid: T_dist in {2.5,5,7.5,10,12.5,15,17.5,20} m  x  T_density in {p00,p50,p70,p80,p90}
      = 40 cells.

NOTE ON COST: lost-ratio here is NOT derivable from cached features alone — the
Phase 1 lost-ratio (lib.lost_ratio) compares baseline detections to detections
RE-RUN on point-pruned LiDAR (run_arm -> FusionRunner.run). So each unique
non-empty removal set needs one 81-frame fusion pass (~0.1 min each; step1.6's
12 arms took 1.2 min total). The 16 step-1.6 arms (p00,p80) are reused from
outputs/step16_preds/*.csv; only the new p50/p70/p90 arms are run.

Output: outputs/step17_cells.csv  (all 40 cells, lost-ratio + % pruned)
"""
import os, sys, csv, json, time
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import (FusionRunner, list_val_samples, paths_for, load_kitti_labels,
                 lost_ratio)
from step15_arms import load_features_csv, run_arm, IOU_THR_PRIMARY, CATS

PHASE1 = "/path/to/RedundancyAD/Multimodal/paper_repro/phase1"
FEAT_CSV  = os.path.join(PHASE1, "outputs/step11_features.csv")
PREDS_DIR = os.path.join(PHASE1, "outputs/step16_preds")   # reuse step1.6 arm cache
OUT_CSV   = os.path.join(PHASE1, "outputs/step17_cells.csv")
os.makedirs(PREDS_DIR, exist_ok=True)

ELIGIBLE = CATS
T_DIST_VALS = [2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0]
# (mode_name, percentile) — mode_name MUST match step1.6 tags for p00/p80 reuse
T_DENS_MODES = [("dist_only", None), ("p50", 50), ("p70", 70), ("p80", 80), ("p90", 90)]


def build_cells(features_by_sid):
    flat = []
    for sid, dets in features_by_sid.items():
        for i, d in enumerate(dets):
            if d.class_name in ELIGIBLE:
                flat.append((sid, i, d.d_ego, d.density, d.class_name))
    cells = []
    for T_dist in T_DIST_VALS:
        gated = [(sid, i, dn, cls) for (sid, i, de, dn, cls) in flat if de <= T_dist]
        gated_dens = np.array([g[2] for g in gated]) if gated else np.array([])
        for mode_name, pct in T_DENS_MODES:
            if pct is None or len(gated_dens) == 0:
                T_dens_val = 0.0
            else:
                T_dens_val = float(np.percentile(gated_dens, pct))
            removal, n_by_cls = set(), defaultdict(int)
            for (sid, i, dn, cls) in gated:
                if dn >= T_dens_val:
                    removal.add((sid, i)); n_by_cls[cls] += 1
            tag = f"Td{T_dist:04.1f}_{mode_name}".replace(".", "p")
            cells.append(dict(tag=tag, T_dist=T_dist, mode=mode_name,
                              T_density_pct=(pct or 0), T_density_val=T_dens_val,
                              n_eligible_in_gate=len(gated), removal=removal,
                              n_removed=len(removal), n_removed_by_cls=dict(n_by_cls)))
    # dedupe by removal-set
    seen = {}
    for c in cells:
        fs = frozenset(c["removal"])
        c["dedup_of"] = seen.get(fs)
        if fs not in seen:
            seen[fs] = c["tag"]
    return cells


def main():
    print(f"[1.7] loading cached Phase 1 features from {FEAT_CSV}")
    features_by_sid = load_features_csv(FEAT_CSV)
    pool_by_cls = {c: sum(1 for v in features_by_sid.values() for d in v if d.class_name == c)
                   for c in ELIGIBLE}
    pool_total = sum(pool_by_cls.values())
    print(f"  {len(features_by_sid)} samples; eligible pool={pool_total} {pool_by_cls}")

    sids = list_val_samples()
    calib_paths = {sid: paths_for(sid)["calib"] for sid in sids}
    gts = {sid: load_kitti_labels(paths_for(sid)["label"]) for sid in sids}

    cells = build_cells(features_by_sid)
    uniq = sum(1 for c in cells if c["dedup_of"] is None)
    print(f"[1.7] {len(cells)} cells; unique removal-sets={uniq}; "
          f"empty-removal={sum(1 for c in cells if c['n_removed']==0)}")

    runner = FusionRunner(erosion=25, depth=20, pca=False)
    baseline = features_by_sid

    # inference per unique non-empty removal set (reuses step16_preds cache)
    preds_by_tag = {}
    t0 = time.time()
    n_cached_hit, n_ran = 0, 0
    for c in cells:
        if c["dedup_of"] is not None:
            continue
        cache = os.path.join(PREDS_DIR, c["tag"] + ".csv")
        if c["n_removed"] == 0:
            preds_by_tag[c["tag"]] = baseline
            continue
        if os.path.exists(cache):
            n_cached_hit += 1
        else:
            n_ran += 1
            print(f"  [run] {c['tag']} (n_removed={c['n_removed']})")
        preds_by_tag[c["tag"]] = run_arm(c["tag"], runner, features_by_sid,
                                         c["removal"], cache)
    print(f"[1.7] inference: {n_cached_hit} cached, {n_ran} newly run, "
          f"{(time.time()-t0)/60:.1f} min")

    # eval each cell (cache per pred-source)
    out_rows, eval_cache = [], {}
    for c in cells:
        src = c["dedup_of"] or c["tag"]
        preds = preds_by_tag[src]
        if src in eval_cache:
            m = eval_cache[src]
        else:
            l50 = lost_ratio(baseline, preds, gts, calib_paths,
                             iou_thr=IOU_THR_PRIMARY, categories=CATS)
            m = dict(lost50_overall=l50["overall"]["lost_ratio"],
                     lost50_Car=l50["Car"]["lost_ratio"],
                     lost50_Ped=l50["Pedestrian"]["lost_ratio"],
                     lost50_Cyc=l50["Cyclist"]["lost_ratio"],
                     n_baseline_TP=l50["overall"]["n_baseline_TP"],
                     n_lost=l50["overall"]["n_lost"])
            eval_cache[src] = m
        nb = c["n_removed_by_cls"]
        out_rows.append(dict(
            tag=c["tag"], T_dist=c["T_dist"], mode=c["mode"],
            T_density_pct=c["T_density_pct"], T_density_val=round(c["T_density_val"], 8),
            n_eligible_in_gate=c["n_eligible_in_gate"], n_pruned=c["n_removed"],
            pct_pruned=c["n_removed"] / pool_total if pool_total else 0.0,
            n_pruned_Car=nb.get("Car", 0), n_pruned_Ped=nb.get("Pedestrian", 0),
            n_pruned_Cyc=nb.get("Cyclist", 0), dedup_of=c["dedup_of"] or "", **m))

    with open(OUT_CSV, "w", newline="") as f:
        fnames = list(out_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fnames); w.writeheader()
        for r in out_rows:
            for k, v in list(r.items()):
                if isinstance(v, float):
                    r[k] = round(v, 6)
            w.writerow(r)
    print(f"[1.7] wrote {len(out_rows)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
