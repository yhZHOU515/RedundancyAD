# ---------------------------------------------------------------------------
# Provenance: bevfusion/holdout/build_holdout_infos.py
# Role: Build the 25-scene val_p01 holdout split + mmdet3d infos (= blob01 (intersection) official val, minus the two mini_val scenes).
#
# REFERENCE / PROVENANCE SCRIPT -- documents how the shipped data/results CSVs
# were produced. NOT runnable from this package alone: it requires the
# excluded raw data (nuScenes), model checkpoints, framework source trees
# (MMDetection3D / BEVFusion, YOLO-LiDAR-Fusion), and the shared geometry/eval
# libraries from the full working tree (lib.py / lib3.py / holdout_lib.py /
# calibration.py). Absolute paths appear as /path/to/... placeholders.
# ---------------------------------------------------------------------------

"""Step 3.9.0 (Option 1 setup) — build mmdet3d infos for val_p01_holdout.

val_p01_holdout = (blob01 ∩ official nuScenes val) MINUS the two mini_val scenes
(scene-0103, scene-0916) = 25 scenes / 995 keyframe samples.

We reuse mmdet3d's own per-sample info builder (_fill_trainval_infos) so the
schema is exactly what BEVFusion expects, then update_pkl_infos converts it to
the v2 (data_list/metainfo) layout. To avoid the converter's whole-dataset scan
(which calls check_file_exist on all 34,149 samples and crashes on the first
non-blob01 file), we first FILTER nusc.sample down to the 25 holdout scenes —
all of whose files are present — so only 995 samples are processed.

Outputs (under data/nuscenes/):
  nuscenes_p01holdout_infos_val.pkl   — 995 samples, v2 schema
Also writes:
  outputs/val_p01_holdout_tokens.txt
  data/nuscenes/nuscenes_p01holdout_smoke_infos_val.pkl + smoke tokens
"""
import os, sys, json, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PHASE3 = "/path/to/RedundancyAD/Multimodal/paper_repro/phase3"
DR = f"{PHASE3}/data/nuscenes"
MMDET3D = f"{PHASE3}/mmdetection3d"
MANIFEST = f"{PHASE3}/outputs/step390_manifest.json"
EXCLUDE = {"scene-0103", "scene-0916"}
PREFIX = "nuscenes_p01holdout"
VERSION = "v1.0-trainval"


def main():
    man = json.load(open(MANIFEST))
    holdout = sorted(s for s in man["val_p01_scene_names"] if s not in EXCLUDE)
    print(f"[infos] holdout={len(holdout)} scenes (excluded {sorted(EXCLUDE)})")
    assert len(holdout) >= 3

    import mmengine
    from nuscenes.nuscenes import NuScenes
    sys.path.insert(0, MMDET3D)
    from tools.dataset_converters.nuscenes_converter import _fill_trainval_infos
    from tools.dataset_converters.update_infos_to_v2 import update_pkl_infos

    print("[infos] loading NuScenes v1.0-trainval…")
    nusc = NuScenes(version=VERSION, dataroot=DR, verbose=False)
    name2token = {s["name"]: s["token"] for s in nusc.scene}
    val_scene_tokens = set(name2token[n] for n in holdout)

    # FILTER to holdout-scene samples only (all files present) before the build.
    all_samples = nusc.sample
    nusc.sample = [s for s in all_samples if s["scene_token"] in val_scene_tokens]
    # CRITICAL: rebuild the token->index map so nusc.get('sample', tok) (used
    # inside get_sample_data/get_boxes) resolves against the filtered list.
    nusc._token2ind["sample"] = {s["token"]: i for i, s in enumerate(nusc.sample)}
    print(f"[infos] filtered nusc.sample {len(all_samples)} -> {len(nusc.sample)} "
          f"(holdout scenes only); rebuilt sample index")

    print("[infos] building infos (val only, max_sweeps=10)…")
    _train, val_infos = _fill_trainval_infos(
        nusc, set(), val_scene_tokens, test=False, max_sweeps=10)
    print(f"[infos] built {len(val_infos)} val infos")

    val_pkl = f"{DR}/{PREFIX}_infos_val.pkl"
    mmengine.dump(dict(infos=val_infos, metadata=dict(version=VERSION)), val_pkl)
    print(f"[infos] dumped old-format pkl; updating to v2…")
    update_pkl_infos("nuscenes", out_dir=DR, pkl_path=val_pkl)

    d = pickle.load(open(val_pkl, "rb"))
    dl = d["data_list"]
    print(f"[infos] v2 val pkl: {len(dl)} samples; metainfo={d['metainfo']}")
    print(f"[infos] sample0 keys: {list(dl[0].keys())}")
    print(f"[infos] sample0 lidar_path: {dl[0]['lidar_points']['lidar_path']}")

    tokens = [s["token"] for s in dl]
    with open(f"{PHASE3}/outputs/val_p01_holdout_tokens.txt", "w") as f:
        f.write("\n".join(tokens) + "\n")
    print(f"[infos] wrote {len(tokens)} holdout tokens")

    smoke = {"metainfo": d["metainfo"], "data_list": dl[:10]}
    pickle.dump(smoke, open(f"{DR}/{PREFIX}_smoke_infos_val.pkl", "wb"))
    with open(f"{PHASE3}/outputs/val_p01_holdout_smoke_tokens.txt", "w") as f:
        f.write("\n".join(s["token"] for s in dl[:10]) + "\n")
    print(f"[infos] wrote smoke pkl (10 samples)")


if __name__ == "__main__":
    main()
