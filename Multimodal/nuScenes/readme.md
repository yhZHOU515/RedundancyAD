# Camera–LiDAR Redundancy Evaluation on nuScenes

The evaluation of image–LiDAR redundancy reveals that high redundancy ratios tend to occur for objects located close to the ego vehicle, where dense LiDAR returns are often observed. In the journal version, this original distance-only analysis serves as the baseline for the distance–density camera–LiDAR evaluation.

The newer distance–density analysis, including the BEVFusion 25-scene nuScenes holdout, matched-threshold grid, regenerated figures, and paper-ready tables, is provided at:

* [Distance–density camera–LiDAR package](../distance_density_extension/)

Related code in this directory could be found at:

* [camera–LiDAR redundancy check](camera-LiDAR.py)
* [redundancy stats check](prune_stats.py)

## Data

For LiDAR and image detection, we use nuScenes-in-KITTI so that the data are compatible with the YOLO-LiDAR-Fusion pipeline.

Raw nuScenes data and pretrained detection checkpoints are not redistributed in this repository because of dataset licensing and storage constraints. Users should obtain the required datasets and models from their official sources.

## Data Quality: Redundancy

For camera–LiDAR data, redundancy can occur when the camera and LiDAR observe the same object. The original distance-only experiment uses object distance as a first-order signal: close-range objects are more likely to be jointly observed by both modalities.

However, the journal analysis shows that distance alone is not sufficient. Some nearby objects may still have weak LiDAR support because of occlusion, object size, viewing angle, or sparse sensor coverage. Therefore, the distance-only rule is extended in the distance–density package by adding an object-level LiDAR support-density gate.

## Task

This experiment investigates redundancy between the front camera and LiDAR.

### (a) Run the full camera–LiDAR fusion detection model

Run the full camera–LiDAR fusion detection model to obtain baseline 3D bounding boxes. The original setup uses the YOLO-LiDAR-Fusion pipeline and compares the full camera–LiDAR fusion detections with detections obtained after pruning selected LiDAR returns.

### (b) Compute each box’s 3D centroid in the LiDAR frame

For each LiDAR-based detection box `b`, compute the 3D centroid of the box and measure its distance from the ego sensor:

`d(b) = ||c(b)||_2`

where `c(b)` is the 3D centroid of the box. Given a distance threshold `T_dist`, the distance-only pruning candidate set is:

`B_pruned = { b in B_LiDAR | d(b) <= T_dist }`

This means that boxes whose centroids lie within the selected distance threshold are removed. By sweeping `T_dist`, the experiment traces how many LiDAR detections are affected and what fraction of baseline true detections is lost.

### (c) Run detection again with pruned LiDAR data

Run detection again after removing the selected LiDAR returns and compare the pruned results with the unpruned baseline. The lost-ratio measures the fraction of baseline true-positive detections that are no longer recovered after pruning:

`lost-ratio = |TP_baseline \ TP_pruned| / |TP_baseline|`

Lower lost-ratio means that pruning better preserves the baseline detections.

## Beyond Distance-Only: Distance–Density Rule

The journal version extends the original distance-only pruning rule with a camera–LiDAR support-density gate:

`rho(b) = n(b) / A_2D(b)`

where `n(b)` is the number of LiDAR returns inside the 3D box and `A_2D(b)` is the projected 2D box area in the image plane. A box is considered a pruning candidate only if it satisfies both gates:

`B_cand = { b in B_LiDAR | d(b) <= T_dist and rho(b) >= T_rho }`

where `T_rho` is a percentile threshold computed among the distance-gated eligible boxes. Setting the density gate to p00 recovers the original distance-only rule.

The full distance–density reproducibility package is available at:

* [Multimodal/distance_density_extension](../distance_density_extension/)

That package includes the aggregate result files, scripts for regenerating paper figures and tables, the complete BEVFusion holdout threshold grid, output figures/tables, and reference provenance code.

## Performance and Goal

The original distance-only analysis shows that object distance is a useful first-order signal for camera–LiDAR redundancy. The t-test result for object distance and cross-modal redundancy (`p = 1.17e-76`) suggests that high cross-modal redundancy is associated with objects close to the ego vehicle.

However, distance-only pruning should be interpreted as a baseline rather than a final criterion. The distance–density rule provides a more selective and controlled redundancy-removal strategy by requiring both proximity and sufficient LiDAR support density. The goal is to remove redundant LiDAR observations while preserving baseline detections as much as possible.
