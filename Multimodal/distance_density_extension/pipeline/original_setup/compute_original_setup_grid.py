# ---------------------------------------------------------------------------
# Provenance: original_camera_lidar_density_gate/step15_arms.py
# Role: Score / remove / re-run / evaluate engine for the original setup (mAP@0.5, lost-ratio per arm).
#
# REFERENCE / PROVENANCE SCRIPT -- documents how the shipped data/results CSVs
# were produced. NOT runnable from this package alone: it requires the
# excluded raw data (nuScenes), model checkpoints, framework source trees
# (MMDetection3D / BEVFusion, YOLO-LiDAR-Fusion), and the shared geometry/eval
# libraries from the full working tree (lib.py / lib3.py / holdout_lib.py /
# calibration.py). Absolute paths appear as /path/to/... placeholders.
# ---------------------------------------------------------------------------

"""Steps 1.2 / 1.3 / 1.4 / 1.5 — score, remove, re-run, evaluate.

Arms (all share frozen Phase 0 eval: 3D-IoU @ 0.5, greedy, mAP@0.5 / R@0.5,
KITTI val 81 samples). Only LiDAR pre-processing differs between arms:

  A baseline    no removal
  B density     remove top-P% objects by density (n / area_2d)
  C random      remove a random P% objects, seeds 0..(n_seeds-1)
  D distance    remove top-P% objects by ego-distance (smallest d_ego first)

For Arm D — "the paper's distance rule" is "remove the closest" because close
objects have the most LiDAR support; the brief phrasing "top P% by ego-distance
(closest objects)" confirms small d_ego = high redundancy under that rule.

Step 1.3 — removal works on the raw velodyne .bin: every point inside any
selected object's OBB is dropped before the fusion model sees the sweep.

Step 1.5 — for each (arm, P, seed): modify velodyne per-frame, re-run fusion,
collect predictions, evaluate, write a row to outputs/step15_arms_results.csv.
The baseline (Arm A) is reused from step11_features.csv — no re-inference.
"""
import os, sys, csv, json, time, tempfile, shutil, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import (FusionRunner, Detection, list_val_samples, paths_for,
                 obb_from_corners, remove_points_for_boxes,
                 load_kitti_labels, eval_arm, lost_ratio, KITTI_ROOT)

PHASE1 = "/path/to/RedundancyAD/Multimodal/paper_repro/phase1"
FEAT_CSV    = os.path.join(PHASE1, "outputs/step11_features.csv")
RESULTS_CSV = os.path.join(PHASE1, "outputs/step15_arms_results.csv")
PREDS_DIR   = os.path.join(PHASE1, "outputs/arm_preds")
os.makedirs(PREDS_DIR, exist_ok=True)

P_LEVELS = [5, 10, 20, 30]
N_SEEDS_RANDOM = 3
IOU_THR_PRIMARY  = 0.5    # frozen in Phase 0
IOU_THR_SECONDARY = 0.25  # diagnostic — see PHASE0_REPORT guess list
CATS = ("Car", "Pedestrian", "Cyclist")


# ---------------------------------------------------------------------------
# Reload feature table → Detection objects
# ---------------------------------------------------------------------------
def load_features_csv(path: str) -> dict:
    """Returns dict {sample_id: [Detection, ...]}."""
    by_sid: dict[str, list[Detection]] = {}
    with open(path) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            corners = np.array([float(x) for x in r["corners_flat"].split(",")],
                               dtype=np.float32).reshape(8, 3)
            d = Detection(
                sample_id=r["sample_id"], class_name=r["class_name"],
                yolo_cls=int(r["yolo_cls"]), conf=float(r["conf"]),
                corners_3d=corners, yaw=float(r["yaw"]),
                centroid=np.array([float(r["cx"]), float(r["cy"]), float(r["cz"])],
                                  dtype=np.float32),
                ground_center=np.array([float(r["gx"]), float(r["gy"]), float(r["gz"])],
                                       dtype=np.float32),
                extent=np.array([float(r["ex"]), float(r["ey"]), float(r["ez"])],
                                dtype=np.float32),
                d_ego=float(r["d_ego"]), n_lidar_pts=int(r["n_lidar_pts"]),
                box_2d_xyxy=np.array([float(r["x1"]), float(r["y1"]),
                                      float(r["x2"]), float(r["y2"])], dtype=np.float32),
                area_2d=float(r["area_2d"]), density=float(r["density"]),
            )
            by_sid.setdefault(d.sample_id, []).append(d)
    return by_sid


# ---------------------------------------------------------------------------
# Selection: which (sample_id, det_idx) get their LiDAR removed
# ---------------------------------------------------------------------------
def select_topP(features_by_sid: dict, score_key: str, P: int,
                seed: int | None = None,
                ascending: bool = False) -> set[tuple]:
    """Flatten all detections, sort by score, take ceil(P/100 * N) of them.

    score_key:
      'density'  → primary redundancy score (higher = more redundant)
      'd_ego'    → distance (closest = most redundant under the paper rule)
                   ascending=True selects closest first
      'random'   → uses seed
    Returns set of (sample_id, index_into_features_by_sid[sid]) tuples.
    """
    flat = []
    for sid, dets in features_by_sid.items():
        for i, d in enumerate(dets):
            flat.append((sid, i, d))
    n = len(flat)
    k = int(np.ceil(P / 100.0 * n))
    if score_key == "random":
        rng = np.random.default_rng(seed)
        order = rng.permutation(n)
        keep = order[:k]
    else:
        scores = np.array([getattr(d, score_key) for (_, _, d) in flat])
        if ascending:
            keep = np.argsort(scores)[:k]
        else:
            keep = np.argsort(-scores)[:k]
    return set((flat[i][0], flat[i][1]) for i in keep)


# ---------------------------------------------------------------------------
# Per-arm runner — modify velodyne per frame, run fusion, collect predictions
# ---------------------------------------------------------------------------
def run_arm(arm_label: str, runner: FusionRunner, features_by_sid: dict,
            removal_set: set[tuple], cache_path: str) -> dict:
    """Returns {sample_id: [Detection, ...]}. Caches to cache_path as CSV."""
    if os.path.exists(cache_path):
        return load_features_csv(cache_path)
    sids = list_val_samples()
    by_sid: dict[str, list[Detection]] = {}
    tmpdir = tempfile.mkdtemp(prefix=f"arm_{arm_label}_")
    t0 = time.time()
    n_removed_total = 0; n_pts_dropped_total = 0
    try:
        for i, sid in enumerate(sids):
            p = paths_for(sid)
            # Build OBBs to remove for this frame.
            obbs = []
            for idx, d in enumerate(features_by_sid.get(sid, [])):
                if (sid, idx) in removal_set:
                    center, R, half_ext = obb_from_corners(d.corners_3d)
                    obbs.append((center, R, half_ext))
            if obbs:
                mod_velo = os.path.join(tmpdir, sid + ".bin")
                stats = remove_points_for_boxes(p["velo"], mod_velo, obbs)
                n_pts_dropped_total += stats["n_removed"]
                velo_to_use = mod_velo
            else:
                velo_to_use = p["velo"]
            dets = runner.run(sid, p["image"], velo_to_use, p["calib"])
            by_sid[sid] = dets
            n_removed_total += len(obbs)
        elapsed = time.time() - t0
        print(f"  [arm {arm_label}] {len(sids)} frames, "
              f"{n_removed_total} obj-removals, {n_pts_dropped_total} pts dropped, "
              f"{elapsed:.1f}s")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    # Cache predictions to CSV so re-runs are free.
    rows = [d.to_row() for sid in sorted(by_sid) for d in by_sid[sid]]
    if rows:
        with open(cache_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    return by_sid


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    print(f"[step1.5] loading features from {FEAT_CSV}")
    features_by_sid = load_features_csv(FEAT_CSV)
    n_det = sum(len(v) for v in features_by_sid.values())
    print(f"  {len(features_by_sid)} samples, {n_det} detections")

    sids = list_val_samples()
    calib_paths = {sid: paths_for(sid)["calib"] for sid in sids}
    gts_per_sample = {sid: load_kitti_labels(paths_for(sid)["label"]) for sid in sids}
    n_gt = sum(len(gts_per_sample[sid][c]) for sid in sids for c in CATS)
    print(f"  GT objects in val (Car+Ped+Cyc): {n_gt}")

    runner = FusionRunner(erosion=25, depth=20, pca=False)

    # Step 1.2 reminder (just printed): redundancy score = density = n / area_2d
    print("[step1.2] redundancy score := n_lidar_pts / area_2d   (higher = more redundant)")

    # Arm A baseline reuses Step 1.1 features (identical to no-removal fusion).
    baseline_preds = features_by_sid

    rows = []  # one per (arm, P, seed)
    # Arm A row (P=0) — no removal, full mAP/Recall reference.
    res50 = eval_arm(baseline_preds, gts_per_sample, calib_paths,
                     iou_thr=IOU_THR_PRIMARY, categories=CATS)
    res25 = eval_arm(baseline_preds, gts_per_sample, calib_paths,
                     iou_thr=IOU_THR_SECONDARY, categories=CATS)
    rows.append(dict(arm="A_baseline", P=0, seed=-1,
                     n_obj_removed=0,
                     **flatten_eval(res50, res25, lost50=None, lost25=None)))

    arms = [
        ("B_density",  "density",  False, [None]),  # primary score, descending
        ("D_distance", "d_ego",    True,  [None]),  # closest first (ascending)
        ("C_random",   "random",   False, list(range(N_SEEDS_RANDOM))),
    ]

    for arm_label, score_key, ascending, seeds in arms:
        for P in P_LEVELS:
            for seed in seeds:
                tag = f"{arm_label}_P{P:02d}"
                if seed is not None:
                    tag = f"{tag}_s{seed}"
                cache = os.path.join(PREDS_DIR, tag + ".csv")
                removal = select_topP(features_by_sid, score_key, P,
                                      seed=seed, ascending=ascending)
                arm_preds = run_arm(tag, runner, features_by_sid, removal, cache)
                res50 = eval_arm(arm_preds, gts_per_sample, calib_paths,
                                 iou_thr=IOU_THR_PRIMARY, categories=CATS)
                res25 = eval_arm(arm_preds, gts_per_sample, calib_paths,
                                 iou_thr=IOU_THR_SECONDARY, categories=CATS)
                lost50 = lost_ratio(baseline_preds, arm_preds, gts_per_sample,
                                    calib_paths, iou_thr=IOU_THR_PRIMARY,
                                    categories=CATS)
                lost25 = lost_ratio(baseline_preds, arm_preds, gts_per_sample,
                                    calib_paths, iou_thr=IOU_THR_SECONDARY,
                                    categories=CATS)
                rows.append(dict(arm=arm_label, P=P, seed=(-1 if seed is None else seed),
                                 n_obj_removed=len(removal),
                                 **flatten_eval(res50, res25, lost50=lost50, lost25=lost25)))
                print(f"    arm={arm_label:11s} P={P:>2d}  seed={seed!s:4s}  "
                      f"mAP50={res50['macro']['mAP']:.4f}  "
                      f"mR50={res50['macro']['mRecall']:.4f}  "
                      f"mAP25={res25['macro']['mAP']:.4f}  "
                      f"lost50={lost50['overall']['lost_ratio']:.4f}  "
                      f"lost25={lost25['overall']['lost_ratio']:.4f}  "
                      f"(removed={len(removal)})")

    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[step1.5] wrote {len(rows)} rows to {RESULTS_CSV}")


def flatten_eval(res50: dict, res25: dict, lost50: dict | None,
                 lost25: dict | None) -> dict:
    flat = {
        "mAP50_macro":    res50["macro"]["mAP"],
        "Recall50_macro": res50["macro"]["mRecall"],
        "mAP25_macro":    res25["macro"]["mAP"],
        "Recall25_macro": res25["macro"]["mRecall"],
    }
    for c in CATS:
        r = res50[c]
        flat[f"AP50_{c}"]     = r["ap"]
        flat[f"Recall50_{c}"] = r["recall"]
        flat[f"TP50_{c}"]     = r["n_TP"]
        flat[f"GT_{c}"]       = r["n_GT"]
        flat[f"pred_{c}"]     = r["n_pred"]
        r25 = res25[c]
        flat[f"AP25_{c}"]     = r25["ap"]
        flat[f"Recall25_{c}"] = r25["recall"]
        flat[f"TP25_{c}"]     = r25["n_TP"]
    flat["lost50_overall"] = lost50["overall"]["lost_ratio"] if lost50 else 0.0
    flat["lost25_overall"] = lost25["overall"]["lost_ratio"] if lost25 else 0.0
    return flat


if __name__ == "__main__":
    main()
