# ---------------------------------------------------------------------------
# Provenance: nuScenes/camera_lidar_exploratory_all_cameras_part1.py
# Role: Diagnostic stage 1: build the per-(object, camera) projection table (camera_lidar_exploratory_stats_all_cameras_part1.csv) over the 94-scene Part 1 partition. Imports the sibling projection module camera_lidar_exploratory_step1.py (not shipped).
#
# REFERENCE / PROVENANCE SCRIPT -- documents how the shipped data/results CSVs
# were produced. NOT runnable from this package alone: it requires the
# excluded raw data (nuScenes), model checkpoints, framework source trees
# (MMDetection3D / BEVFusion, YOLO-LiDAR-Fusion), and the shared geometry/eval
# libraries from the full working tree (lib.py / lib3.py / holdout_lib.py /
# calibration.py). Absolute paths appear as /path/to/... placeholders.
# ---------------------------------------------------------------------------

"""
camera_lidar_exploratory_all_cameras_part1.py

Part 1 all-cameras stats CSV for nuScenes v1.0-trainval, restricted to the
fully-present scenes found by part1_data_verification.py (the 94 scenes whose
keyframe files were extracted from v1.0-trainval01_blobs).

It reuses the EXACT projection logic, visibility filters, ego-distance, and
in-box LiDAR counting from camera_lidar_exploratory_step1.py by importing that
module (no duplicated geometry). The only differences from step5 are:
  - nuScenes version is v1.0-trainval (not mini),
  - iteration is restricted to the Part 1 scene tokens (keyframes only), and
  - two extra columns (scene_token, scene_name) are carried through so the
    downstream scene-level split + sidecars need no re-derivation.

Does NOT train, prune, or create YOLO labels. Reads nuScenes only (no writes to
the dataset).
"""

import matplotlib
matplotlib.use("Agg")

import os
import numpy as np
import pandas as pd

from nuscenes.nuscenes import NuScenes

# Reuse step-1's frame transforms, LiDAR counting, projection + filters.
import camera_lidar_exploratory_step1 as s1


# =====================================================================
# CONFIGURATION
# =====================================================================
NUSCENES_DATAROOT = "/path/to/nuscenes"
NUSCENES_VERSION = "v1.0-trainval"

# Scene tokens to process (the 94 fully-present Part 1 scenes from verification).
SCENE_TOKENS_FILE = "part1_data_verification_outputs/part1_scene_tokens.txt"

OUTPUT_CSV = "camera_lidar_exploratory_stats_all_cameras_part1.csv"

CAMERA_CHANNELS = [
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
]
LIDAR_CHANNEL = s1.LIDAR_CHANNEL

# step-1 columns + two Part-1 carry-through columns.
CSV_COLUMNS = ["scene_token", "scene_name"] + s1.CSV_COLUMNS


def load_scene_tokens():
    with open(SCENE_TOKENS_FILE) as f:
        return [ln.strip() for ln in f if ln.strip()]


def iter_scene_samples(nusc, scene_token):
    """Yield every keyframe (sample) of a scene in temporal order."""
    scene = nusc.get("scene", scene_token)
    sample_token = scene["first_sample_token"]
    while sample_token:
        sample = nusc.get("sample", sample_token)
        yield sample
        sample_token = sample["next"]


def process(nusc, scene_tokens):
    rows = []
    global_sample_index = 0
    for si, scene_token in enumerate(scene_tokens):
        scene = nusc.get("scene", scene_token)
        scene_name = scene["name"]

        for sample in iter_scene_samples(nusc, scene_token):
            lidar_sd_token = sample["data"][LIDAR_CHANNEL]

            # Camera-independent quantities, computed once per sample.
            lidar_counts = s1.count_lidar_points_in_box(nusc, sample)
            ann_info = {}
            for ann_token in sample["anns"]:
                ann_rec = nusc.get("sample_annotation", ann_token)
                ego_box = s1.get_box_in_ego_frame(nusc, ann_token, lidar_sd_token)
                ann_info[ann_token] = {
                    "category_name": ann_rec["category_name"],
                    "instance_token": ann_rec["instance_token"],
                    "visibility_token": ann_rec.get("visibility_token", ""),
                    "distance_m": float(np.linalg.norm(ego_box.center)),
                }

            for cam in CAMERA_CHANNELS:
                if cam not in sample["data"]:
                    continue
                cam_sd_token = sample["data"][cam]
                cam_sd = nusc.get("sample_data", cam_sd_token)
                iw, ih = cam_sd["width"], cam_sd["height"]

                for ann_token in sample["anns"]:
                    visible, bbox = s1.project_box_to_camera(
                        nusc, ann_token, cam_sd_token, iw, ih
                    )
                    if not visible:
                        continue
                    x1, y1, x2, y2 = bbox
                    info = ann_info[ann_token]
                    rows.append({
                        "scene_token": scene_token,
                        "scene_name": scene_name,
                        "sample_token": sample["token"],
                        "sample_index": global_sample_index,
                        "annotation_token": ann_token,
                        "instance_token": info["instance_token"],
                        "category_name": info["category_name"],
                        "distance_m": info["distance_m"],
                        "lidar_point_count": lidar_counts[ann_token],
                        "camera_name": cam,
                        "bbox_x1": x1, "bbox_y1": y1,
                        "bbox_x2": x2, "bbox_y2": y2,
                        "image_width": iw, "image_height": ih,
                        "visibility_token": info["visibility_token"],
                    })
            global_sample_index += 1

        print(f"[{si + 1}/{len(scene_tokens)}] {scene_name} "
              f"({scene_token[:8]}): cumulative rows={len(rows)}", flush=True)

    return pd.DataFrame(rows, columns=CSV_COLUMNS)


def main():
    print("================ CONFIGURATION ================")
    print(f"NUSCENES_DATAROOT : {NUSCENES_DATAROOT}")
    print(f"NUSCENES_VERSION  : {NUSCENES_VERSION}")
    print(f"SCENE_TOKENS_FILE : {SCENE_TOKENS_FILE}")
    print(f"CAMERAS           : {', '.join(CAMERA_CHANNELS)}")
    print(f"LIDAR_CHANNEL     : {LIDAR_CHANNEL}")
    print(f"OUTPUT_CSV        : {OUTPUT_CSV}")
    print("===============================================\n", flush=True)

    scene_tokens = load_scene_tokens()
    print(f"Loaded {len(scene_tokens)} Part 1 scene tokens.\n", flush=True)

    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_DATAROOT, verbose=False)
    df = process(nusc, scene_tokens)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n================ STATS ================")
    print(f"scenes processed : {len(scene_tokens)}")
    print(f"keyframes        : {df['sample_token'].nunique()}")
    print(f"total rows       : {len(df)}")
    if len(df):
        lp = df["lidar_point_count"]
        print(f"distance_m       : min={df['distance_m'].min():.2f} "
              f"mean={df['distance_m'].mean():.2f} max={df['distance_m'].max():.2f}")
        print(f"lidar_point_count: min={int(lp.min())} mean={lp.mean():.2f} "
              f"max={int(lp.max())}")
        print(f"zero-LiDAR rows  : {int((lp == 0).sum())} ({100.0 * (lp == 0).mean():.1f}%)")
        print("\nrows per camera:")
        pc = df["camera_name"].value_counts()
        for cam in CAMERA_CHANNELS:
            print(f"  {cam:<16}: {int(pc.get(cam, 0))}")
        print(f"\ncategories       : {df['category_name'].nunique()}")
    print("=======================================")
    print(f"\nSaved: {os.path.abspath(OUTPUT_CSV)}")


if __name__ == "__main__":
    main()
