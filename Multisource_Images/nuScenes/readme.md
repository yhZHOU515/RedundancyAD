This experiment aims to define and quantify the redundancy at the instance level in the context of object detection and explore the effects of removing redundancy in multisource images.

[Example code](/pair2-deduplication-and-training.ipynb)

# Data

The full [nuScenes](https://www.nuscenes.org/nuscenes) dataset was released in 2019 with all 1,000 scenes. The full dataset includes approximately 1.4M camera images, 390k LIDAR sweeps, 1.4M RADAR sweeps, 
and 1.4M object bounding boxes in 40k keyframes. There are ground truth labels for 23 object classes. The nuScenes-mini dataset used in this experiment is a smaller subset containing 10 scenes and 404 frames 
with the same sensor setting and annotation structure.

## Sensor setup

It includes 360-degree sensor coverage with data collected from six cameras, five radars, and one LiDAR sensor, along with high-definition maps


<img width="500" height="350" alt="sensor" src="https://github.com/user-attachments/assets/eed14043-5c5b-461a-abf4-34f202dd5b52" />


<img width="350" height="300" alt="camera setting" src="https://github.com/user-attachments/assets/e8beb981-5deb-4bf8-9f3f-1c09ca02d590" />


# Data Quality

This case study evaluates the **Redundancy** dimension to illustrate the relationship between the layers in the framework.

The sensor setup above shows that there are six pairs of overlapping fields of view (FoV) in the cameras. 
These overlapping parts indicate areas where cross-camera redundancy may occur, and these six pairs are our research focus.


### (a) Identify the overlapping FoV based on the nuScenes dataset sensor setting:

- **Pair 1**: `CAM_FRONT` and `CAM_FRONT_RIGHT` overlap by 15°  
- **Pair 2**: `CAM_FRONT` and `CAM_FRONT_LEFT` overlap by 15°  
- **Pair 3**: `CAM_FRONT_RIGHT` and `CAM_BACK_RIGHT` overlap by 15°  
- **Pair 4**: `CAM_FRONT_LEFT` and `CAM_BACK_LEFT` overlap by 15°  
- **Pair 5**: The rear camera `CAM_BACK` overlaps with `CAM_BACK_RIGHT` by 20°  
- **Pair 6**: The rear camera `CAM_BACK` overlaps with `CAM_BACK_LEFT` by 20°  

---

### (b) Crop images based on overlapped angles

For example, redundancy occurs in the bottom two cropped images.


<img width="1272" height="710" alt="cropping" src="https://github.com/user-attachments/assets/da873b78-a818-4a57-8a9e-78ac2c87d446" />



---

### (c) Calculate cosine similarity

For the cropped images in the six pairs, we calculate the cosine similarity of each sample and exclude the ones without redundant instances. This provides us with a preliminary understanding of redundancy.

---

### (d) Create different levels of redundant training datasets

To investigate how redundancy affects inference performance, for each pair of overlapping detections, 
we compute a **Bounding Box Completeness Score (BCS)**, which indicates how completely the bounding box (BBox) presents the instance.

Let `BBox_full` be the original (uncropped) 2D bounding-box area in the image, and let `BBox_clipped` be the visible portion after clipping to the image boundaries. We define:

<img width="300" height="50" alt="BCS" src="https://github.com/user-attachments/assets/0a878032-fb5f-43c0-bbad-fcd2237a3c1f" />


where `b` indexes a candidate box. Within each redundant group, if


<img width="300" height="45" alt="rule" src="https://github.com/user-attachments/assets/07fd0d6c-ee22-4cc8-8786-06a9df310de1" />

we retain only the box with the higher BCS and discard the lower one; otherwise, we preserve both boxes. As the threshold `τ_BCS` increases, fewer boxes are removed, thus retaining more redundancy while still preferring more complete annotations.

# Task and Application

We take object detection as the task in this case study, and use [YOLO v8](https://yolov8.com/) as the model. Model performance is evaluated using mean average precision (mAP).
mAP50 evaluates how accurately bounding boxes align with ground-truth annotations with at least 50% overlap.

Our baseline detection model was evaluated on 1,401 images containing 12,286 object instances across 14 categories of nuScenes. 
Overall, the detector achieved a box precision of 0.78, a recall of 0.63, and an mAP50 of 0.72.

---

### (e) Train on different levels of redundant datasets

Train Yolov8 on each pair of overlapping camera images: from the previous step, we obtain training sets with different levels of redundancy. 
Next, we train the model using these training datasets and evaluate how removing redundancy affects the inference performance. 


# Performance and Goal

For each pair, the thresholds are set from 0.0 to 1.0, with a 0.2 interval. As the threshold increases, the instances in the training dataset are kept more, meaning the redundancy is reserved more, till the 1.0 threshold keeps all the instances for training. 
Interestingly, the trends reveal that a **less redundant** training dataset can **achieve or even surpass** the performance level of using the full training dataset.  


<img width="400" height="600" alt="output" src="https://github.com/user-attachments/assets/5920be22-b2b3-4380-8ed1-8b9f5e7a93b4" />

