The evaluation of image-LiDAR redundancy reveals that high redundancy ratios tend to occur for objects located close to the ego-vehicle, where dense LiDAR returns.

Related code could be found at: 
- [camera-LiDAR redundancy check](./camera-LiDAR.py)
- [redundancy stats check](./prune_stats.py)
- [camera-LiDAR fusion detection model repo](https://github.com/TimKie/YOLO-LiDAR-Fusion/tree/main)

# Data

For LiDAR and image detection, we use [nuScenes-in-KITTI](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/scripts/export_kitti.py) in order to be compatible with the fusion model using KITTI format.

# Data Quality: Redundancy

For example, when the weather conditions are good and clear, there is less need to keep high-overlapping information from LiDAR and cameras. The models trained on such data would have little information gain and performance improvement.

This experiment investigates the redundancy between the front camera and LiDAR: 

![Sample](https://github.com/user-attachments/assets/2b7e3901-921b-4757-b973-0ade2b37f5f8)


# Task

### (a) Run the full camera-LiDAR fusion detection model

Run the full [camera-LiDAR fusion detection model](https://github.com/TimKie/YOLO-LiDAR-Fusion/tree/main) to get baseline 3D bounding boxes. 
Check the overlapping detections between this and the detection using only LiDAR data as redundancy. 
Let  `B_LiDAR` be the set of boxes detected using LiDAR only, and `B_base` the set detected by the image-LiDAR fusion. The *redundancy ratio* for one frame is

> RR = |{b ∈ B_base | ∃ b' ∈ B_LiDAR : IoU(b, b') ≥ θ}| / |B_base|

### (b) Compute each box’s 3D centroid in the LiDAR frame

Compute each box’s 3D centroid in the LiDAR frame, and measure its distance from the ego sensor:

> c(b) = (1/8) * Σ(v_i) for i = 1 to 8

and

> d(b) = ||c(b)||_2


where `{v_i}` are the eight corner vertices of `b`. We then set a single pruning threshold `T_dist` (meters). All boxes whose centroids lie within this distance are removed:


> B_pruned = { B_LiDAR | d(B_LiDAR) ≥ T_dist }


By sweeping `T_dist`, we could trace how many boxes are pruned and what fraction of true detections is lost, 
thus choosing a trade-off point that maximally reduces redundancy without harming detection performance too much. This distance threshold is chosen based on statistical results.


### (c) Run LiDAR detection again with pruned data

Run LiDAR detection again using the partially removed LiDAR data and observe how removing redundancy affects the performance. 
Let `B_pruned` be the set after pruning. We define the *lost ratio* `l` as the fraction of baseline boxes removed due to removing the redundancy:


> l = |B_base \ B_pruned| / |B_base|


# Performance and Goal
T-test results of the object in LiDAR distance threshold and cross-modal redundancy (p-value=1.17e-76) suggest that the high cross-modal redundancy objects remain very close to the ego-vehicle. 
This supports us in removing the close-range LiDAR data as redundancy. The detection performance can be viewed as unaffected by choosing a reasonable threshold and removing near-range LiDAR points. 
It also proves that in many scenes, **close-range LiDAR data and camera data** are redundant. While removing them could have little impact, the efficiency can be improved due to the decrease in data points to be processed. 


<img width="400" height="350" alt="distance_pruning_curve" src="https://github.com/user-attachments/assets/ad2a291b-2cc1-40d9-aad7-ca4f65606dfd" />


