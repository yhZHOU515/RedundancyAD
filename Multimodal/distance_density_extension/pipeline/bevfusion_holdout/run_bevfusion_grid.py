# ---------------------------------------------------------------------------
# Provenance: bevfusion/holdout/run_density_sensitivity_grid.py
# Role: Run the 5x5 distance x density grid (BEVFusion inference per cell) -> data/results/holdout_grid_5x5.csv.
#
# REFERENCE / PROVENANCE SCRIPT -- documents how the shipped data/results CSVs
# were produced. NOT runnable from this package alone: it requires the
# excluded raw data (nuScenes), model checkpoints, framework source trees
# (MMDetection3D / BEVFusion, YOLO-LiDAR-Fusion), and the shared geometry/eval
# libraries from the full working tree (lib.py / lib3.py / holdout_lib.py /
# calibration.py). Absolute paths appear as /path/to/... placeholders.
# ---------------------------------------------------------------------------

"""Step 3.12 — density-gate sensitivity sweep on val_p01_holdout.

5 T_dist x 5 T_density = 25 cells. Reuses cached cells from step39_cells.csv;
runs ONLY the missing 13. Same machinery as step394_optionB.py (same ckpt/config,
DontCare-excluded pool, per-bin density percentiles recomputed within the
distance-gated subset, same keyframe LiDAR-swap removal, same dual eval).

Output: outputs/step312_cells.csv  (all 25 cells, merged)
"""
import os, sys, csv, json, time, pickle
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/path/to/RedundancyAD/Multimodal/paper_repro/phase1")
sys.path.insert(0, "/path/to/RedundancyAD/Multimodal/YOLO-LiDAR-Fusion/Code")

from holdout_lib import (run_bevfusion_inference, native_nuscenes_eval_custom,
                         scene_names_for_tokens, DR)
from lib3 import obb_from_corners
from step34_eval import (load_predictions_in_lidar_frame, load_gt_lidar,
                         evaluate, lost_ratio)

PHASE3 = "/path/to/RedundancyAD/Multimodal/paper_repro/phase3"
VERSION = "v1.0-trainval"
FULL_ANN = "nuscenes_p01holdout_infos_val.pkl"
INFOS_PKL = f"{DR}/{FULL_ANN}"
TOKENS_F = f"{PHASE3}/outputs/val_p01_holdout_tokens.txt"
FEATS = f"{PHASE3}/outputs/step39_features_holdout.csv"
BASELINE_PRED = f"{PHASE3}/outputs/holdout_baseline/preds/pred_instances_3d/results_nusc.json"
PREDS_DIR = f"{PHASE3}/outputs/holdout_cells"      # same dir as step39 (reuse preds)
STEP39_CSV = f"{PHASE3}/outputs/step39_cells.csv"
OUT_CSV = f"{PHASE3}/outputs/step312_cells.csv"
ELIGIBLE = ("Car", "Pedestrian", "Cyclist")
os.makedirs(PREDS_DIR, exist_ok=True)

T_DIST = [10.0, 15.0, 20.0, 22.5, 30.0]
T_DENS = [0, 50, 70, 80, 90]
GRID = [(t, p) for t in T_DIST for p in T_DENS]


def tag_for(T, p):
    return f"Td{T:04.1f}_p{p:02d}".replace(".", "p")


def corners_from_row(r):
    return np.array([float(x) for x in r["corners_flat"].split(",")]).reshape(8, 3)


def build_removal(rows, T_dist, p):
    gated = [i for i, r in enumerate(rows)
             if r["class_phase1"] in ELIGIBLE and float(r["d_ego"]) <= T_dist]
    if p == 0:
        return set(gated), 0.0, len(gated)
    dens = np.array([float(rows[i]["density"]) for i in gated])
    tval = float(np.percentile(dens, p)) if len(dens) else 0.0
    removal = {i for i in gated if float(rows[i]["density"]) >= tval}
    return removal, tval, len(gated)


def main():
    # ---- which cells already exist in step39_cells.csv ----
    cached = {}
    for r in csv.DictReader(open(STEP39_CSV)):
        if r["tag"] == "baseline":
            continue
        cached[(float(r["T_dist"]), int(r["T_density_pct"]))] = r
    missing = [(t, p) for (t, p) in GRID if (t, p) not in cached]
    n_grid_cached = sum(1 for c in GRID if c in cached)
    print(f"[3.12] grid={len(GRID)} cached={n_grid_cached} to-run={len(missing)}")
    print("[3.12] missing:", [tag_for(t, p) for t, p in missing])

    out_rows = []
    # carry over cached rows for grid cells (keep their full schema)
    for (t, p) in GRID:
        if (t, p) in cached:
            out_rows.append(dict(cached[(t, p)], pred_source="cached_step39"))

    if missing:
        from nuscenes.nuscenes import NuScenes
        nusc = NuScenes(version=VERSION, dataroot=DR, verbose=False)
        tokens = open(TOKENS_F).read().split()
        rows = list(csv.DictReader(open(FEATS)))
        infos = pickle.load(open(INFOS_PKL, "rb"))
        lidar_fn = {s["token"]: s["lidar_points"]["lidar_path"] for s in infos["data_list"]}
        scenes = scene_names_for_tokens(nusc, tokens)
        pop = {c: sum(1 for r in rows if r["class_phase1"] == c) for c in ELIGIBLE}
        pool = sum(pop.values())
        print(f"[3.12] {len(rows)} dets; {len(tokens)} tokens; {len(scenes)} scenes; pool={pool}")

        gts = load_gt_lidar(nusc, tokens)
        # baseline lost-ratio reference
        base_preds = load_predictions_in_lidar_frame(BASELINE_PRED, nusc, tokens)
        _, _, _, b50m = evaluate(base_preds, gts, 0.5)

        t_start = time.time()
        for k, (T, p) in enumerate(missing):
            tag = tag_for(T, p)
            removal, tval, n_gated = build_removal(rows, T, p)
            by_bin = defaultdict(list)
            for i in removal:
                r = rows[i]
                c, R, h = obb_from_corners(corners_from_row(r))
                by_bin[lidar_fn[r["sample_id"]]].append((c, R, h))
            t1 = time.time()
            pj = run_bevfusion_inference(FULL_ANN, f"{PREDS_DIR}/{tag}",
                                         removal_by_bin=dict(by_bin))
            if pj is None:
                print(f"[3.12] {tag}: FAILED inference"); continue
            arm_preds = load_predictions_in_lidar_frame(pj, nusc, tokens)
            r50, mAP50, mR50, m50 = evaluate(arm_preds, gts, 0.5)
            l50_per, l50_ov = lost_ratio(b50m, m50)
            nm, nds, per = native_nuscenes_eval_custom(
                nusc, pj, tokens, f"{PHASE3}/outputs/holdout_native/{tag}", scenes)
            nrc = defaultdict(int)
            for i in removal:
                nrc[rows[i]["class_phase1"]] += 1
            out_rows.append(dict(
                tag=tag, T_dist=T, T_density_pct=p, T_density_val=round(tval, 5),
                n_removed=len(removal), removal_pct_of_pool=round(len(removal) / pool, 5),
                n_removed_Car=nrc["Car"], n_removed_Ped=nrc["Pedestrian"], n_removed_Cyc=nrc["Cyclist"],
                mAP50=round(mAP50, 5), mR50=round(mR50, 5), lost50_overall=round(l50_ov, 5),
                native_mAP=round(nm, 5), native_NDS=round(nds, 5),
                AP50_Car=round(r50["Car"]["ap"], 5), AP50_Ped=round(r50["Pedestrian"]["ap"], 5),
                AP50_Cyc=round(r50["Cyclist"]["ap"], 5),
                lost50_Ped=round(l50_per["Pedestrian"]["lost_ratio"], 5),
                pred_source="step312_new"))
            print(f"[3.12] {tag}: n_removed={len(removal)} mAP50={mAP50:.4f} nat={nm:.4f} "
                  f"PedAP={r50['Pedestrian']['ap']:.4f} lost={l50_ov:.4f} "
                  f"({time.time()-t1:.0f}s, {k+1}/{len(missing)})")
        print(f"[3.12] ran {len(missing)} cells in {(time.time()-t_start)/60:.1f} min")

    # ---- write merged 25-cell csv (sorted by T_dist then T_density) ----
    out_rows.sort(key=lambda r: (float(r["T_dist"]), int(r["T_density_pct"])))
    # union of all keys (cached rows may have extra columns)
    fld = ["tag", "T_dist", "T_density_pct", "T_density_val", "n_removed",
           "removal_pct_of_pool", "n_removed_Car", "n_removed_Ped", "n_removed_Cyc",
           "mAP50", "mR50", "lost50_overall", "native_mAP", "native_NDS",
           "AP50_Car", "AP50_Ped", "AP50_Cyc", "lost50_Ped", "pred_source"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fld, extrasaction="ignore")
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"[3.12] wrote {len(out_rows)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
