# ---------------------------------------------------------------------------
# Provenance: bevfusion/holdout/extract_holdout_features.py
# Role: Build the per-object eligible-pool feature table (74,464 Car/Pedestrian/Cyclist objects) from the holdout baseline predictions.
#
# REFERENCE / PROVENANCE SCRIPT -- documents how the shipped data/results CSVs
# were produced. NOT runnable from this package alone: it requires the
# excluded raw data (nuScenes), model checkpoints, framework source trees
# (MMDetection3D / BEVFusion, YOLO-LiDAR-Fusion), and the shared geometry/eval
# libraries from the full working tree (lib.py / lib3.py / holdout_lib.py /
# calibration.py). Absolute paths appear as /path/to/... placeholders.
# ---------------------------------------------------------------------------

"""Step 3.9 — per-object feature table for the val_p01_holdout baseline preds.

Identical feature definitions to Step 3.1 (lib3 helpers) but on the
val_p01_holdout set: nuScenes loaded as v1.0-trainval, the holdout baseline
predictions, and the holdout infos pkl (for per-camera intrinsics/extrinsics).

Run AFTER the holdout baseline inference produces results_nusc.json.
"""
import os, sys, json, csv, pickle, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib3 import (Detection3, CAMERAS, NUSC_TO_PHASE1,
                  transform_box_global_to_lidar, corners_from_box_lidar,
                  obb_from_corners, points_in_obb,
                  project_through_all_cameras, gt_lidar_per_sample)
from step31_features import best_gt_visibility

PHASE3 = "/path/to/RedundancyAD/Multimodal/paper_repro/phase3"
DATAROOT = f"{PHASE3}/data/nuscenes"
VERSION = "v1.0-trainval"
PRED_JSON = f"{PHASE3}/outputs/holdout_baseline/preds/pred_instances_3d/results_nusc.json"
INFOS_PKL = f"{DATAROOT}/nuscenes_p01holdout_infos_val.pkl"
TOKEN_FILE = f"{PHASE3}/outputs/val_p01_holdout_tokens.txt"
OUT_CSV = f"{PHASE3}/outputs/step39_features_holdout.csv"


def main():
    from nuscenes.nuscenes import NuScenes
    print("[step3.9-feat] loading nuScenes v1.0-trainval…")
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)
    raw = json.load(open(PRED_JSON))
    infos = pickle.load(open(INFOS_PKL, "rb"))
    tokens = open(TOKEN_FILE).read().split()
    info_by_token = {s["token"]: s for s in infos["data_list"]}
    print(f"[step3.9-feat] {len(raw['results'])} pred-tokens; {len(tokens)} holdout tokens; "
          f"{len(info_by_token)} infos-tokens")

    rows = []
    t0 = time.time()
    for i, tok in enumerate(tokens):
        if tok not in raw["results"] or tok not in info_by_token:
            continue
        sample_info = info_by_token[tok]
        lidar_path = os.path.join(DATAROOT, "samples", "LIDAR_TOP",
                                  sample_info["lidar_points"]["lidar_path"])
        full_pts = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3].astype(np.float64)
        sample = nusc.get("sample", tok)
        sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        ep = nusc.get("ego_pose", sd["ego_pose_token"])
        gts = gt_lidar_per_sample(nusc, tok)
        for d in raw["results"][tok]:
            center_lidar, yaw_lidar = transform_box_global_to_lidar(
                d["translation"], d["rotation"],
                ep["translation"], ep["rotation"],
                cs["translation"], cs["rotation"])
            size_wlh = np.asarray(d["size"], dtype=np.float64)
            corners = corners_from_box_lidar(center_lidar, size_wlh, yaw_lidar)
            obb_c, obb_R, obb_half = obb_from_corners(corners)
            n_pts = int(points_in_obb(full_pts, obb_c, obb_R, obb_half).sum())
            cam_results, primary = project_through_all_cameras(corners, sample_info)
            if primary is None:
                primary_str = ""; xyxy = np.zeros(4); area = 0.0
            else:
                primary_str = primary; xyxy, area, _ = cam_results[primary]
            cams_visible = tuple(c for c in CAMERAS if cam_results[c][0] is not None)
            density = n_pts / area if area > 0 else 0.0
            centroid = corners.mean(axis=0)
            d_ego = float(np.linalg.norm(centroid))
            ground = corners[np.argsort(corners[:, 2])[:4]].mean(axis=0)
            class_p1 = NUSC_TO_PHASE1.get(d["detection_name"], "DontCare")
            vis = best_gt_visibility(corners, gts)
            det = Detection3(
                sample_id=tok, class_name=d["detection_name"],
                class_phase1=class_p1, conf=float(d["detection_score"]),
                corners_3d=corners.astype(np.float32),
                centroid=centroid.astype(np.float32),
                ground_center=ground.astype(np.float32),
                size_wlh=size_wlh.astype(np.float32), yaw=float(yaw_lidar),
                d_ego=d_ego, n_lidar_pts=n_pts, primary_cam=primary_str,
                box_2d_xyxy=xyxy.astype(np.float32),
                area_2d=area, density=float(density),
                cams_visible=cams_visible, visibility_token=vis)
            rows.append(det.to_row())
        if (i + 1) % 100 == 0:
            print(f"  [{i+1:4d}/{len(tokens)}] dets={len(rows)} elapsed={time.time()-t0:.1f}s")

    fieldnames = list(rows[0].keys())
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)
    print(f"\n[step3.9-feat] wrote {len(rows)} rows to {OUT_CSV}")
    from collections import Counter
    print("[step3.9-feat] class_phase1:", dict(Counter(r["class_phase1"] for r in rows)))


if __name__ == "__main__":
    main()
