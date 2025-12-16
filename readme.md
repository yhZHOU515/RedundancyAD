# Paper Introduction
## Background

Next-generation autonomous vehicles (AVs) rely on large volumes of multisource and multimodal ($M^2$) data to support frequent real-time decision-making. 
In practice, the quality of such data varies due to environmental conditions, sensor limitations, and system noise. 
Despite its critical role, data quality (DQ) remains largely underexplored in the AV literature, which predominantly focuses on model and algorithmic advances. 
This work addresses this gap by studying redundancy as a fundamental DQ issue in AV datasets.

## Methods
We model and measure redundancy in both multisource image data and multimodal image–LiDAR data using the [nuScenes](https://www.nuscenes.org/) dataset and [Argoverse 2](https://www.argoverse.org/av2.html) dataset. 
Controlled redundancy removal is applied to curated subsets of the data, and its impact is evaluated on the YOLOv8 object detection task. 
Cross-modal analysis is further conducted to examine redundancy-related DQ issues between image and LiDAR modalities.

## Results
Experimental results show that partially removing redundancy from multisource image data can improve object detection performance. 
Multimodal analysis further reveals substantial redundancy between image and LiDAR data.

## Conclusion / Implications.
This study demonstrates that redundancy is a measurable and actionable DQ factor with direct implications for AV perception performance. 
By data-centric evaluation, this work highlights critical but underexplored challenges at the intersection of data quality, task orchestration, and system performance. 
The findings provide practical guidance for developing more adaptive, explainable, and resilient AV systems that can better handle heterogeneous and dynamic data streams.

# Research Design

## Illustration of multisource and multimodal data in autonomous vehicles (AVs)

<img width="3000" height="1688" alt="source and modal" src="https://github.com/user-attachments/assets/39dddffa-865f-4db9-8790-42cf30bb4487" />

## Research questions

- RQ1: What redundancy exists in multisource and multimodal AV data?
- RQ2: What redundancy should we remove, and how should we remove it?
- RQ3: How does redundancy removal affect object detection model performance?

<img width="2798" height="1339" alt="research design" src="https://github.com/user-attachments/assets/9adfe7f3-bc81-4aca-8a57-f34c256c8ca0" />



