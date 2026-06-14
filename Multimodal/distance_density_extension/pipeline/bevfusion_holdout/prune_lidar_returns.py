# ---------------------------------------------------------------------------
# Provenance: bevfusion/holdout/holdout_lib.py
# Role: LiDAR-return pruning machinery (remove_points_5d + keyframe LiDAR-swap removal) plus the BEVFusion inference / native-eval helpers the grid driver calls.
#
# REFERENCE / PROVENANCE SCRIPT -- documents how the shipped data/results CSVs
# were produced. NOT runnable from this package alone: it requires the
# excluded raw data (nuScenes), model checkpoints, framework source trees
# (MMDetection3D / BEVFusion, YOLO-LiDAR-Fusion), and the shared geometry/eval
# libraries from the full working tree (lib.py / lib3.py / holdout_lib.py /
# calibration.py). Absolute paths appear as /path/to/... placeholders.
# ---------------------------------------------------------------------------

"""Shared helpers for the val_p01_holdout experiment (Step 3.9).

Reuses Phase 3 machinery verbatim:
  - BEVFusion inference via tools/test.py (same CONFIG, same CKPT), only the
    test ann_file is overridden to the holdout infos pkl.
  - Same keyframe-only LiDAR swap removal mechanism (remove_points_5d).
  - native nuScenes eval via a CUSTOM SPLIT: we patch create_splits_scenes so
    eval_set='val' maps to the holdout scenes, and patch load_gt to restrict GT
    to the evaluated tokens (so it also works on a sub-scene smoke set).
"""
import os, sys, json, shutil, subprocess, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib3 import remove_points_5d

PHASE3 = "/path/to/RedundancyAD/Multimodal/paper_repro/phase3"
DR = f"{PHASE3}/data/nuscenes"
MMDET3D = f"{PHASE3}/mmdetection3d"
LIDAR_DIR = f"{DR}/samples/LIDAR_TOP"          # swap dir (per-file symlinks)
LIDAR_SRC = "/path/to/nuscenes/samples/LIDAR_TOP"
CONFIG = "projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py"
CKPT = f"{PHASE3}/checkpoints/bevfusion_lidar-cam_voxel0075_nus.pth"
VENV = f"{PHASE3}/.venv/bin/python"


def run_bevfusion_inference(ann_rel, work_dir, removal_by_bin=None, timeout=7200):
    """Run BEVFusion test on the samples in ann_rel (relative to data_root).
    removal_by_bin: {lidar_bin_filename: [(center,R,half_extent), ...]} to delete
    points from those keyframes (swap-and-restore). Returns pred_json path or None.
    """
    pred_json = f"{work_dir}/preds/pred_instances_3d/results_nusc.json"
    if os.path.exists(pred_json):
        return pred_json
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    swapped = []
    tmp_dir = None
    try:
        if removal_by_bin:
            tmp_dir = f"/tmp/holdout_{os.path.basename(work_dir)}"
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
            os.makedirs(tmp_dir)
            for bin_name, obbs in removal_by_bin.items():
                orig = f"{LIDAR_SRC}/{bin_name}"
                mod = f"{tmp_dir}/{bin_name}"
                remove_points_5d(orig, mod, obbs)
                sym = f"{LIDAR_DIR}/{bin_name}"
                tgt = os.readlink(sym)
                os.unlink(sym); os.symlink(mod, sym)
                swapped.append((sym, tgt))
        cmd = [
            VENV, "tools/test.py", CONFIG, CKPT, "--work-dir", work_dir,
            "--cfg-options", "test_evaluator.format_only=True",
            f"test_evaluator.jsonfile_prefix={work_dir}/preds",
            f"test_dataloader.dataset.ann_file={ann_rel}",
            f"test_evaluator.ann_file={DR}/{ann_rel}",
        ]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["PYTHONPATH"] = "."
        proc = subprocess.run(cmd, cwd=MMDET3D, env=env, capture_output=True,
                              text=True, timeout=timeout)
        if proc.returncode != 0:
            print(f"  [infer] test.py FAILED (exit {proc.returncode}):\n"
                  f"{proc.stderr[-2000:]}")
            return None
        return pred_json if os.path.exists(pred_json) else None
    finally:
        for sym, tgt in swapped:
            if os.path.lexists(sym):
                os.unlink(sym)
            os.symlink(tgt, sym)
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def scene_names_for_tokens(nusc, tokens):
    names = set()
    for t in tokens:
        sc = nusc.get("scene", nusc.get("sample", t)["scene_token"])
        names.add(sc["name"])
    return sorted(names)


def native_nuscenes_eval_custom(nusc, pred_json, tokens, work_dir, scene_names):
    """Native nuScenes detection eval restricted to `tokens` via a custom split.
    Returns (mean_ap, nd_score, per_class_mean_ap_dict)."""
    raw = json.load(open(pred_json))
    filt = {"meta": raw["meta"],
            "results": {t: raw["results"].get(t, []) for t in tokens}}
    os.makedirs(work_dir, exist_ok=True)
    fp = os.path.join(work_dir, "preds_filtered.json")
    json.dump(filt, open(fp, "w"))

    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.evaluate import NuScenesEval
    import nuscenes.eval.common.loaders as L
    import nuscenes.eval.detection.evaluate as E
    from nuscenes.utils.splits import create_splits_scenes as real_css
    from nuscenes.eval.common.data_classes import EvalBoxes

    tset = set(tokens)
    orig_css = L.create_splits_scenes
    orig_load_gt = E.load_gt

    def patched_css(verbose=False):
        d = real_css()
        d["val"] = list(scene_names)   # override the (allowed) 'val' split
        return d

    def patched_load_gt(nusc_, eval_split, box_cls, verbose=False):
        gt = orig_load_gt(nusc_, eval_split, box_cls, verbose=verbose)
        out = EvalBoxes()
        for st in gt.sample_tokens:
            if st in tset:
                out.add_boxes(st, gt.boxes[st])
        return out

    L.create_splits_scenes = patched_css
    E.load_gt = patched_load_gt
    try:
        cfg = config_factory("detection_cvpr_2019")
        out_dir = os.path.join(work_dir, "nusc_eval")
        os.makedirs(out_dir, exist_ok=True)
        ev = NuScenesEval(nusc, config=cfg, result_path=fp, eval_set="val",
                          output_dir=out_dir, verbose=False)
        metrics = ev.main(render_curves=False)
    finally:
        L.create_splits_scenes = orig_css
        E.load_gt = orig_load_gt
    per_class = metrics.get("mean_dist_aps", {})
    return float(metrics["mean_ap"]), float(metrics["nd_score"]), per_class
