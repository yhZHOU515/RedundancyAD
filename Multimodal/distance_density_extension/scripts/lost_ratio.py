"""Lost-ratio metric (Eq. 11) — standalone reference implementation.

The *lost-ratio* is the headline retention metric used throughout the
journal-extension distance--density experiments. It answers: of the baseline
(no-removal) true-positive detections, what fraction is *no longer* recovered
after a pruning arm is applied?

    lost-ratio  =  | TP_baseline \\ TP_arm |  /  | TP_baseline |          (Eq. 11)

where TP_baseline / TP_arm are the sets of ground-truth boxes matched at
IoU >= `iou_thr` by the baseline / arm predictions respectively. A GT box is
identified by ``(sample_id, gt_index)``; "lost" boxes are those the baseline
matched but the arm did not.

This module reproduces *verbatim* the verified core used in the shipped
experiment scripts:

  - BEVFusion holdout / pilot:  ``bevfusion/.../step34_eval.py::lost_ratio``
  - Original conference setup:   ``original_camera_lidar_density_gate/lib.py::lost_ratio``

Both compute the identical set-difference ratio; they differ only in how the
matched-TP sets are produced (a greedy, score-sorted, 3D-IoU matcher run once
for the baseline and once for the arm). That matcher is dataset/framework
specific (it needs the 3D-IoU helper, calibration, and per-frame predictions
that are intentionally NOT shipped in this package), so the runnable
surface here is the metric itself, given already-matched sets.

No numbers are computed or stored here; this is the metric definition only.
"""
from __future__ import annotations

CATEGORIES = ("Car", "Pedestrian", "Cyclist")


def lost_ratio(baseline_matched: dict, arm_matched: dict,
               categories=CATEGORIES) -> tuple[dict, float]:
    """Per-class and overall lost-ratio from already-matched TP sets.

    Parameters
    ----------
    baseline_matched, arm_matched
        ``{class_name: set_of_matched_GT_keys}``. Each key uniquely identifies a
        ground-truth box (e.g. ``(sample_token, gt_index)``). These are the
        outputs of running the greedy IoU matcher on the baseline and arm
        predictions respectively.
    categories
        Classes to aggregate over (default: Car / Pedestrian / Cyclist).

    Returns
    -------
    (per_class, overall)
        ``per_class[c] = {"n_baseline_TP", "n_lost", "lost_ratio"}`` and
        ``overall`` is the micro-averaged lost-ratio over all classes.

    This is the exact computation from ``step34_eval.py`` (BEVFusion) and the
    aggregation in ``lib.py`` (original setup), kept framework-independent.
    """
    total_b = 0
    total_lost = 0
    per: dict[str, dict] = {}
    for c in categories:
        b = baseline_matched[c]
        a = arm_matched[c]
        lost = b - a                      # baseline TPs the arm failed to recover
        n_b = len(b)
        per[c] = dict(
            n_baseline_TP=n_b,
            n_lost=len(lost),
            lost_ratio=len(lost) / n_b if n_b else 0.0,
        )
        total_b += n_b
        total_lost += len(lost)
    overall = total_lost / total_b if total_b else 0.0
    return per, overall


if __name__ == "__main__":
    # Tiny self-check on synthetic matched sets (illustrative only — not paper data).
    base = {
        "Car":        {("s0", 0), ("s0", 1), ("s1", 0)},
        "Pedestrian": {("s0", 5)},
        "Cyclist":    set(),
    }
    arm = {
        "Car":        {("s0", 0), ("s1", 0)},     # lost ("s0", 1)
        "Pedestrian": {("s0", 5)},                # nothing lost
        "Cyclist":    set(),
    }
    per, overall = lost_ratio(base, arm)
    for c, v in per.items():
        print(f"{c:11s} lost_ratio={v['lost_ratio']:.4f}  "
              f"(lost {v['n_lost']}/{v['n_baseline_TP']})")
    print(f"overall     lost_ratio={overall:.4f}")
