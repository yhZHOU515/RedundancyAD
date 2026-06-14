# ---------------------------------------------------------------------------
# Provenance: bevfusion/holdout/step34_eval.py
# Role: Compute per-cell metrics: 3D-IoU mAP@0.5, lost-ratio (Eq. 11), and native nuScenes mAP/NDS.
#
# REFERENCE / PROVENANCE SCRIPT -- documents how the shipped data/results CSVs
# were produced. NOT runnable from this package alone: it requires the
# excluded raw data (nuScenes), model checkpoints, framework source trees
# (MMDetection3D / BEVFusion, YOLO-LiDAR-Fusion), and the shared geometry/eval
# libraries from the full working tree (lib.py / lib3.py / holdout_lib.py /
# calibration.py). Absolute paths appear as /path/to/... placeholders.
# ---------------------------------------------------------------------------

"""Step 3.4 — Evaluate all arms identically.

For each (arm, P, seed) in arm_preds/manifest.json plus the baseline:
  - apples-to-apples 3D-IoU mAP @ 0.5 and 0.25 (the same iou_3d helper Phase 1
    used, after transforming preds and GT into LIDAR_TOP frame)
  - lost-ratio at IoU 0.5 and 0.25 vs the A_baseline predictions
  - native nuScenes mAP @ center-distance {0.5,1,2,4 m} + NDS via nuscenes-devkit

Output: outputs/step34_results.csv with one row per (arm, P, seed).
"""
import os, sys, json, csv, pickle, time
import numpy as np
from pyquaternion import Quaternion

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/path/to/RedundancyAD/Multimodal/paper_repro/phase1")
sys.path.insert(0, "/path/to/RedundancyAD/Multimodal/YOLO-LiDAR-Fusion/Code")
from lib3 import (transform_box_global_to_lidar, NUSC_TO_PHASE1,
                  PHASE1_BUCKETS, DATAROOT, VERSION)
from lib import iou_3d  # Phase-1's helper, ties to upstream get_axis_aligned_bbox

PHASE3 = "/path/to/RedundancyAD/Multimodal/paper_repro/phase3"
BASELINE_PRED = f"{PHASE3}/outputs/bevfusion_lcam/preds/pred_instances_3d/results_nusc.json"
MANIFEST = f"{PHASE3}/outputs/arm_preds/manifest.json"
TOKEN_FILE = f"{PHASE3}/mini_val_tokens.txt"
OUT_CSV = f"{PHASE3}/outputs/step34_results.csv"


def voc_ap_11pt(rec, prec):
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = prec[rec >= t].max() if (rec >= t).any() else 0.0
        ap += p / 11.0
    return float(ap)


def load_predictions_in_lidar_frame(json_path, nusc, tokens):
    """Return {tok: [(class_phase1, score, center_lidar, size_wlh, yaw_lidar), ...]}."""
    raw = json.load(open(json_path))
    out = {}
    for tok in tokens:
        if tok not in raw["results"]:
            out[tok] = []
            continue
        sample = nusc.get("sample", tok)
        sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        ep = nusc.get("ego_pose", sd["ego_pose_token"])
        local = []
        for d in raw["results"][tok]:
            bucket = NUSC_TO_PHASE1.get(d["detection_name"])
            if bucket not in PHASE1_BUCKETS:
                continue
            cl, yaw = transform_box_global_to_lidar(
                d["translation"], d["rotation"],
                ep["translation"], ep["rotation"],
                cs["translation"], cs["rotation"],
            )
            local.append((bucket, float(d["detection_score"]), cl, d["size"], yaw))
        out[tok] = local
    return out


def load_gt_lidar(nusc, tokens):
    """Return {tok: {bucket: [(c_lidar, size_hwl_for_upstream, yaw), ...]}}."""
    out = {}
    for tok in tokens:
        sample = nusc.get("sample", tok)
        sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        ep = nusc.get("ego_pose", sd["ego_pose_token"])
        gt = {c: [] for c in PHASE1_BUCKETS}
        for at in sample["anns"]:
            ann = nusc.get("sample_annotation", at)
            cat = ann["category_name"]
            if cat.startswith("vehicle.car"):
                bucket = "Car"
            elif cat.startswith("human.pedestrian"):
                bucket = "Pedestrian"
            elif cat == "vehicle.bicycle":
                bucket = "Cyclist"
            else:
                continue
            cl, yaw = transform_box_global_to_lidar(
                ann["translation"], ann["rotation"],
                ep["translation"], ep["rotation"],
                cs["translation"], cs["rotation"],
            )
            # upstream expects (h, w, l); nuScenes ann size = (w, l, h)
            gt[bucket].append((cl, [ann["size"][2], ann["size"][0], ann["size"][1]], yaw))
        out[tok] = gt
    return out


def evaluate(preds_lidar, gts_lidar, thr=0.5):
    per_cls = {c: dict(ap=0.0, recall=0.0, n_TP=0, n_GT=0, n_pred=0)
               for c in PHASE1_BUCKETS}
    flat = {c: [] for c in PHASE1_BUCKETS}
    for tok, ps in preds_lidar.items():
        for bucket, score, cl, size_wlh, yaw in ps:
            flat[bucket].append((tok, score, cl, size_wlh, yaw))
    used = {c: {tok: np.zeros(len(gts_lidar[tok][c]), dtype=bool)
                for tok in gts_lidar} for c in PHASE1_BUCKETS}
    matched_idx_by_bucket = {c: set() for c in PHASE1_BUCKETS}  # (tok, gt_idx)
    for c in PHASE1_BUCKETS:
        preds = sorted(flat[c], key=lambda x: -x[1])
        n_gt = sum(len(gts_lidar[tok][c]) for tok in gts_lidar)
        per_cls[c]["n_GT"] = n_gt; per_cls[c]["n_pred"] = len(preds)
        tp = np.zeros(len(preds)); fp = np.zeros(len(preds))
        for i, (tok, _, cl, sz_wlh, yaw) in enumerate(preds):
            pred_size = [sz_wlh[2], sz_wlh[0], sz_wlh[1]]
            entries = gts_lidar.get(tok, {}).get(c, [])
            usedm = used[c][tok]
            best_iou = 0.0; best_j = -1
            for j, (gc, gs, gry) in enumerate(entries):
                if usedm[j]: continue
                iou = iou_3d(cl, pred_size, yaw, gc, gs, gry)
                if iou > best_iou:
                    best_iou = iou; best_j = j
            if best_iou >= thr and best_j >= 0:
                tp[i] = 1.0; usedm[best_j] = True
                matched_idx_by_bucket[c].add((tok, best_j))
            else:
                fp[i] = 1.0
        tp_cum = np.cumsum(tp); fp_cum = np.cumsum(fp)
        rec = tp_cum / max(n_gt, 1)
        prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-6)
        per_cls[c]["ap"] = voc_ap_11pt(rec, prec) if n_gt > 0 else 0.0
        per_cls[c]["n_TP"] = int(tp.sum())
        per_cls[c]["recall"] = float(tp.sum() / n_gt) if n_gt > 0 else 0.0
    macro_ap = float(np.mean([per_cls[c]["ap"] for c in PHASE1_BUCKETS]))
    macro_r  = float(np.mean([per_cls[c]["recall"] for c in PHASE1_BUCKETS]))
    return per_cls, macro_ap, macro_r, matched_idx_by_bucket


def lost_ratio(baseline_matched, arm_matched):
    total_b = 0; total_lost = 0
    per = {}
    for c in PHASE1_BUCKETS:
        b = baseline_matched[c]
        a = arm_matched[c]
        lost = b - a
        per[c] = dict(n_baseline_TP=len(b), n_lost=len(lost),
                      lost_ratio=len(lost) / len(b) if b else 0.0)
        total_b += len(b); total_lost += len(lost)
    overall = total_lost / total_b if total_b else 0.0
    return per, overall


def native_nuscenes_eval(nusc, pred_json, tokens, work_dir):
    """Return (mAP, NDS) on mini_val for this prediction file."""
    raw = json.load(open(pred_json))
    filt = {"meta": raw["meta"],
            "results": {t: raw["results"].get(t, []) for t in tokens}}
    os.makedirs(work_dir, exist_ok=True)
    fp = os.path.join(work_dir, "preds_filtered.json")
    json.dump(filt, open(fp, "w"))
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.evaluate import NuScenesEval
    cfg = config_factory("detection_cvpr_2019")
    out_dir = os.path.join(work_dir, "nusc_eval")
    os.makedirs(out_dir, exist_ok=True)
    ev = NuScenesEval(nusc, config=cfg, result_path=fp,
                      eval_set="mini_val", output_dir=out_dir, verbose=False)
    metrics = ev.main(render_curves=False)
    return float(metrics["mean_ap"]), float(metrics["nd_score"])


def main():
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)
    tokens = open(TOKEN_FILE).read().split()
    print(f"[step3.4] {len(tokens)} mini_val tokens")
    print("[step3.4] loading GT in LIDAR frame…")
    gts_lidar = load_gt_lidar(nusc, tokens)

    # Baseline
    print("[step3.4] evaluating A_baseline…")
    baseline_preds = load_predictions_in_lidar_frame(BASELINE_PRED, nusc, tokens)
    base50, base50_mAP, base50_mR, base50_matched = evaluate(baseline_preds, gts_lidar, 0.5)
    base25, base25_mAP, base25_mR, base25_matched = evaluate(baseline_preds, gts_lidar, 0.25)
    native_work = f"{PHASE3}/outputs/native_eval_A_baseline"
    base_native_mAP, base_native_NDS = native_nuscenes_eval(nusc, BASELINE_PRED, tokens, native_work)
    print(f"  A_baseline  mAP50={base50_mAP:.4f}  mR50={base50_mR:.4f}  "
          f"mAP25={base25_mAP:.4f}  native_mAP={base_native_mAP:.4f}  NDS={base_native_NDS:.4f}")

    manifest = json.load(open(MANIFEST))
    print(f"[step3.4] {len(manifest)} arm runs to evaluate")
    rows = []
    # Baseline row
    rows.append(dict(arm="A_baseline", P=0, seed=-1, n_obj_removed=0,
                     mAP50=base50_mAP, mR50=base50_mR,
                     mAP25=base25_mAP, mR25=base25_mR,
                     lost50_overall=0.0, lost25_overall=0.0,
                     native_mAP=base_native_mAP, native_NDS=base_native_NDS,
                     AP50_Car=base50["Car"]["ap"], AP50_Ped=base50["Pedestrian"]["ap"],
                     AP50_Cyc=base50["Cyclist"]["ap"],
                     AP25_Car=base25["Car"]["ap"], AP25_Ped=base25["Pedestrian"]["ap"],
                     AP25_Cyc=base25["Cyclist"]["ap"],
                     TP50_Car=base50["Car"]["n_TP"], TP50_Ped=base50["Pedestrian"]["n_TP"],
                     TP50_Cyc=base50["Cyclist"]["n_TP"]))

    for m in manifest:
        tag = m["tag"]
        print(f"  evaluating {tag}…")
        arm_preds = load_predictions_in_lidar_frame(m["pred_json"], nusc, tokens)
        r50, mAP50, mR50, r50_matched = evaluate(arm_preds, gts_lidar, 0.5)
        r25, mAP25, mR25, r25_matched = evaluate(arm_preds, gts_lidar, 0.25)
        l50_per, l50_ov = lost_ratio(base50_matched, r50_matched)
        l25_per, l25_ov = lost_ratio(base25_matched, r25_matched)
        nw = f"{PHASE3}/outputs/native_eval_{tag}"
        try:
            nm, nds = native_nuscenes_eval(nusc, m["pred_json"], tokens, nw)
        except Exception as e:
            print(f"    native eval failed: {e}")
            nm, nds = 0.0, 0.0
        rows.append(dict(arm=m["arm"], P=m["P"], seed=m["seed"],
                         n_obj_removed=m["n_obj_removed"],
                         mAP50=mAP50, mR50=mR50, mAP25=mAP25, mR25=mR25,
                         lost50_overall=l50_ov, lost25_overall=l25_ov,
                         native_mAP=nm, native_NDS=nds,
                         AP50_Car=r50["Car"]["ap"], AP50_Ped=r50["Pedestrian"]["ap"],
                         AP50_Cyc=r50["Cyclist"]["ap"],
                         AP25_Car=r25["Car"]["ap"], AP25_Ped=r25["Pedestrian"]["ap"],
                         AP25_Cyc=r25["Cyclist"]["ap"],
                         TP50_Car=r50["Car"]["n_TP"], TP50_Ped=r50["Pedestrian"]["n_TP"],
                         TP50_Cyc=r50["Cyclist"]["n_TP"]))

    fieldnames = list(rows[0].keys())
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            for k, v in list(r.items()):
                if isinstance(v, float):
                    r[k] = round(v, 4)
        w.writerows(rows)
    print(f"\n[step3.4] wrote {len(rows)} rows → {OUT_CSV}")


if __name__ == "__main__":
    main()
