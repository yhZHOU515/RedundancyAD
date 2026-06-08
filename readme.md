## Paper Introduction

### Background

Next-generation autonomous vehicles (AVs) rely on large volumes of multisource and multimodal M² data to support real-time decision-making. In practice, data quality (DQ) varies across sources and modalities due to environmental conditions and sensor limitations, yet AV research has largely prioritized algorithm design over DQ analysis. This work focuses on redundancy as a fundamental but underexplored DQ issue in AV datasets.

### Methods

We model and measure redundancy in multisource camera data and multimodal image–LiDAR data using the nuScenes and Argoverse 2 (AV2) datasets. For multisource camera redundancy, we study overlapping camera fields of view and use the Bounding Box Completeness Score (BCS) to retain the more complete camera observation. For camera–LiDAR redundancy, we first evaluate the original distance-only pruning rule and then extend it to a conjunctive distance–density criterion that combines object distance with LiDAR support density. The multisource experiments train and evaluate YOLOv8 on BCS-pruned camera datasets, while the distance–density multimodal experiments evaluate a pretrained BEVFusion model on density-pruned LiDAR inputs.

### Results

The results show that selectively removing redundant multisource image labels can preserve, and in some settings improve, object detection performance. In nuScenes, mAP50 improves from 0.66 to 0.70, from 0.64 to 0.67, and from 0.53 to 0.55 on representative overlap regions, while other camera pairs remain close to or above their baselines under selected pruning thresholds. In AV2, 4.1–8.6% of labels are removed, and mAP50 remains near the 0.64 baseline.

For multimodal redundancy, the distance–density rule removes redundant LiDAR observations more selectively than the previous distance-only rule while better preserving BEVFusion detections on a 25-scene nuScenes holdout. At T_dist = 30 m, distance-only pruning removes 58% of eligible objects and yields a lost-ratio of 0.104, whereas distance–density pruning removes only 12%/6% under the p80/p90 density gates and reduces the lost-ratio to 0.065/0.050. It also preserves higher detection performance, maintaining mAP50 around 0.37–0.38, compared with 0.34 for distance-only pruning.

### Conclusion / Implications

This study shows that redundancy is a measurable and actionable DQ factor in autonomous driving datasets. By evaluating redundancy from a data-centric perspective, the work shows how selective redundancy removal can affect dataset efficiency and downstream perception performance across multisource camera data and multimodal camera–LiDAR data.

## Research Design

The study evaluates redundancy in two settings:

* **Multisource camera redundancy:** overlapping camera views may contain duplicated observations of the same physical object. BCS-guided pruning retains the more complete camera observation.
* **Multimodal camera–LiDAR redundancy:** camera and LiDAR may observe the same object. The distance-only LiDAR pruning rule is extended to a distance–density criterion using object distance and LiDAR support density.

## Research Questions

* **RQ1:** How to define and model redundancy in multisource and multimodal AV data for object detection?
* **RQ2:** How can redundancy be measured, and what evidence identifies redundant observations that can be removed while preserving detection performance?
* **RQ3:** How does redundancy removal affect object detection performance across datasets, sensing modalities, and detection models?


<img width="2798" height="1339" alt="research design" src="https://github.com/user-attachments/assets/9adfe7f3-bc81-4aca-8a57-f34c256c8ca0" />





