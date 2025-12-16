import os
import cv2
import numpy as np
import open3d as o3d
from ultralytics import YOLO
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from sklearn.cluster import DBSCAN

# Modules from the Visual-Sensor-Fusion repository.
import Utils as ut
import Lidar2Camera as l2c  # This is the KITTI-based LiDAR2Camera class.
import LidarUtils as lu
import FusionUtils as fu
import YoloUtils as yu  # Optional, for drawing utilities

############################################################
# Helper: Compute Sensor-to-Global Transform
############################################################
def sensor_to_global_transform(nusc, sd_rec):
    """
    Returns a 4x4 homogeneous transform that maps points 
    from the sensor coordinate frame to the global frame.
    sd_rec: a 'sample_data' record for the sensor.
    """
    # 1) Sensor-to-ego transformation using the calibrated sensor record.
    cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    sensor2ego_rot = Quaternion(cs_rec['rotation']).rotation_matrix
    sensor2ego_trans = np.array(cs_rec['translation'])
    T_sensor2ego = np.eye(4)
    T_sensor2ego[:3, :3] = sensor2ego_rot
    T_sensor2ego[:3, 3] = sensor2ego_trans

    # 2) Ego-to-global transformation.
    ego_pose = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    ego2global_rot = Quaternion(ego_pose['rotation']).rotation_matrix
    ego2global_trans = np.array(ego_pose['translation'])
    T_ego2global = np.eye(4)
    T_ego2global[:3, :3] = ego2global_rot
    T_ego2global[:3, 3] = ego2global_trans

    # Combined sensor-to-global transformation.
    T_sensor2global = T_ego2global @ T_sensor2ego
    return T_sensor2global

############################################################
# Helper: Compute LiDAR-to-Camera Transform (nuScenes version)
############################################################
def get_lidar_to_cam_transform(nusc, cam_sd, lidar_sd):
    """
    Computes a 4x4 transform that takes points from the LiDAR frame
    (in nuScenes coordinates) to the Camera frame (in nuScenes coordinates),
    using sensor-to-global transforms.
    """
    T_cam2global = sensor_to_global_transform(nusc, cam_sd)
    T_lidar2global = sensor_to_global_transform(nusc, lidar_sd)
    T_cam_lidar = np.linalg.inv(T_cam2global) @ T_lidar2global
    return T_cam_lidar

############################################################
# Helper: Convert nuScenes LiDAR to KITTI LiDAR Coordinates
############################################################
def get_lidar_to_cam_transform_kitti(nusc, cam_sd, lidar_sd):
    """
    Computes the LiDAR-to-Camera transform and applies a fixed rotation
    to convert nuScenes LiDAR coordinates (x forward, y right, z down)
    to KITTI coordinates (x forward, y left, z up).
    """
    # First, get the nuScenes LiDAR-to-Camera transform.
    T_cam_lidar = get_lidar_to_cam_transform(nusc, cam_sd, lidar_sd)
    
    # Fixed conversion: Flip the y and z axes.
    R_nusc2kitti = np.eye(4)
    R_nusc2kitti[1, 1] = -1  # Flip y.
    R_nusc2kitti[2, 2] = -1  # Flip z.
    
    # Apply the conversion.
    T_cam_lidar_kitti = T_cam_lidar @ R_nusc2kitti
    return T_cam_lidar_kitti

############################################################
# Helper: Create KITTI-Style Calibration Dictionary
############################################################
def create_kitti_calibration(nusc, cam_sd, lidar_sd):
    """
    Converts nuScenes calibration information into a KITTI-style calibration
    dictionary. This includes:
      - P2: (3x4) Camera projection matrix.
      - Tr_velo_to_cam: (3x4) LiDAR-to-camera transform.
      - R0_rect: (3x3) Rectification matrix.
    """
    # Retrieve the camera intrinsics.
    cs_cam = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
    K = np.array(cs_cam['camera_intrinsic']).reshape(3, 3)
    P2 = np.hstack([K, np.zeros((3, 1))])
    
    # Compute the LiDAR-to-camera transform with coordinate conversion.
    T_cam_lidar = get_lidar_to_cam_transform_kitti(nusc, cam_sd, lidar_sd)
    
    calib_dict = {
        "P2": P2.flatten(),
        "Tr_velo_to_cam": T_cam_lidar[:3, :].flatten(),  # Use top 3 rows.
        "R0_rect": np.eye(3).flatten()  # Identity rectification.
    }
    return calib_dict

import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix

def create_kitti_calibration_from_devkit(nusc, cam_sd, lidar_sd):
    """
    Creates a KITTI-style calibration dictionary using the dev-kit transformation logic.
    
    This function:
    1. Retrieves the calibrated sensor records for camera and LiDAR.
    2. Computes the sensor-to-ego (and ego-to-camera) transforms.
    3. Computes the LiDAR-to-camera transform.
    4. Applies the fixed rotation (kitti_to_nu_lidar) to convert from nuScenes LiDAR
       (x: forward, y: right, z: down) to KITTI LiDAR (x: forward, y: left, z: up).
    5. Constructs the KITTI calibration dictionary.
    """
    # Retrieve the camera and LiDAR calibrated sensor records.
    cs_record_cam = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
    cs_record_lid = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
    
    # Compute the LiDAR-to-ego transform (lidar frame to ego vehicle frame).
    lid_to_ego = transform_matrix(cs_record_lid['translation'],
                                  Quaternion(cs_record_lid['rotation']),
                                  inverse=False)
    
    # Compute the ego-to-camera transform (ego frame to camera frame).
    ego_to_cam = transform_matrix(cs_record_cam['translation'],
                                  Quaternion(cs_record_cam['rotation']),
                                  inverse=True)
    
    # LiDAR-to-camera transform (in nuScenes coordinates).
    velo_to_cam = np.dot(ego_to_cam, lid_to_ego)
    
    # Apply fixed rotation to convert from nuScenes LiDAR coordinates to KITTI.
    # In nuScenes: x-forward, y-right, z-down. In KITTI: x-forward, y-left, z-up.
    # The conversion is achieved with a rotation of pi/2 about the z-axis.
    kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
    velo_to_cam_kitti = np.dot(velo_to_cam, kitti_to_nu_lidar.transformation_matrix)
    
    # Get camera intrinsics.
    K = np.array(cs_record_cam['camera_intrinsic']).reshape(3, 3)
    P2 = np.hstack([K, np.zeros((3, 1))])  # KITTI projection matrix format.
    
    # Build calibration dictionary.
    calib_dict = {
        "P2": P2.flatten(),
        "Tr_velo_to_cam": velo_to_cam_kitti[:3, :].flatten(),  # Use top 3 rows.
        "R0_rect": np.eye(3).flatten()  # Cameras are already rectified.
    }
    return calib_dict


############################################################
# Helper: Load LiDAR Points (x,y,z only)
############################################################
def load_lidar_points(filepath):
    """
    Loads LiDAR points from a file.
    For .bin or .pcd.bin files (e.g., in nuScenes), each point has 5 values.
    This function returns only the first 3 (x, y, z).
    """
    if filepath.endswith('.bin') or filepath.endswith('.pcd.bin'):
        points = np.fromfile(filepath, dtype=np.float32).reshape(-1, 5)
        return points[:, :3]
    else:
        pcd = o3d.io.read_point_cloud(filepath)
        return np.asarray(pcd.points)

############################################################
# Helper: 3D Clustering (DBSCAN) to get 3D bounding boxes
############################################################
def cluster_lidar_points_3d(points_3d, eps=10, min_samples=20):
    """
    Cluster 3D LiDAR points using DBSCAN.
    Returns a list of 3D bounding boxes (each as 
      [x_min, y_min, z_min, x_max, y_max, z_max]).
    """
    if points_3d.size == 0:
        return []
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points_3d)
    labels = clustering.labels_
    unique_labels = set(labels)
    boxes_3d = []
    for label in unique_labels:
        if label == -1:
            continue  # Skip noise.
        cluster_pts = points_3d[labels == label]
        x_min, y_min, z_min = np.min(cluster_pts, axis=0)
        x_max, y_max, z_max = np.max(cluster_pts, axis=0)
        boxes_3d.append([x_min, y_min, z_min, x_max, y_max, z_max])
    return boxes_3d

############################################################
# Helper: Project a 3D Box to 2D Image
############################################################
def project_3d_box_to_image(lidar2cam, box_3d, image_shape):
    """
    Given a 3D bounding box in LiDAR coordinates [x_min, y_min, z_min, x_max, y_max, z_max],
    compute the 8 corners, project each corner into the image, and return the 2D bounding box
    (u_min, v_min, u_max, v_max) that encloses the projected corners.
    """
    x_min, y_min, z_min, x_max, y_max, z_max = box_3d
    # Define 8 corners of the 3D box.
    corners = np.array([
        [x_min, y_min, z_min],
        [x_min, y_min, z_max],
        [x_min, y_max, z_min],
        [x_min, y_max, z_max],
        [x_max, y_min, z_min],
        [x_max, y_min, z_max],
        [x_max, y_max, z_min],
        [x_max, y_max, z_max]
    ])
    
    projected = []
    for corner in corners:
        corner_hom = np.hstack([corner, 1])
        # Transform using LiDAR2Camera's V2C and R0 matrices.
        cam_point = np.dot(lidar2cam.V2C, corner_hom)
        rect_point = np.dot(lidar2cam.R0, cam_point)
        rect_point_hom = np.hstack([rect_point, 1])
        proj = np.dot(lidar2cam.P, rect_point_hom)
        if proj[2] <= 0:
            continue  # Skip points behind the camera.
        u = proj[0] / proj[2]
        v = proj[1] / proj[2]
        projected.append([u, v])
    
    if not projected:
        return None
    projected = np.array(projected)
    u_min = np.min(projected[:, 0])
    v_min = np.min(projected[:, 1])
    u_max = np.max(projected[:, 0])
    v_max = np.max(projected[:, 1])
    
    # Clip to image size.
    width, height = image_shape
    u_min = max(0, min(u_min, width))
    u_max = max(0, min(u_max, width))
    v_min = max(0, min(v_min, height))
    v_max = max(0, min(v_max, height))
    return np.array([u_min, v_min, u_max, v_max])

############################################################
# Main Fusion Function: mid_level_fusion_nuscenes
############################################################
def mid_level_fusion_nuscenes(nusc, sample_token, cam_sensor, lidar_sensor, yolo_model,
                              display_image=True, save_image=False):
    # Retrieve sample data.
    sample = nusc.get('sample', sample_token)
    cam_sd = nusc.get('sample_data', sample['data'][cam_sensor])
    lidar_sd = nusc.get('sample_data', sample['data'][lidar_sensor])
    
    # Load the camera image.
    image_path = os.path.join(nusc.dataroot, cam_sd['filename'])
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image at", image_path)
        return None, 0
    
    # Convert nuScenes calibration to KITTI-style and create LiDAR2Camera object.
    # Replace this line:
    # calib_kitti = create_kitti_calibration(nusc, cam_sd, lidar_sd)

   # With the new version:
    calib_kitti = create_kitti_calibration_from_devkit(nusc, cam_sd, lidar_sd)

    lidar2cam = l2c.LiDAR2Camera(calib_kitti)
    
    # --- Step 1: YOLOv8 Object Detection on Image ---
    results = yolo_model.predict(image, conf=0.4)
    dets = []
    for box in results[0].boxes:
        xyxy = box.xyxy.cpu().numpy().flatten().tolist()  # [x1, y1, x2, y2]
        x1, y1, x2, y2 = xyxy
        conf = float(box.conf.cpu().numpy()[0])
        cls_id = int(box.cls.cpu().numpy()[0])
        w = x2 - x1
        h = y2 - y1
        dets.append([cls_id, (x1, y1, w, h), conf])
    detections = np.array(dets, dtype=object)
    
    # --- Step 2: Load LiDAR Points and Cluster in 3D ---
    lidar_path = os.path.join(nusc.dataroot, lidar_sd['filename'])
    point_cloud = load_lidar_points(lidar_path)  # (N,3)
    boxes_3d = cluster_lidar_points_3d(point_cloud, eps=5, min_samples=20)
    
    # --- Step 3: Project Each 3D Box to 2D ---
    lidar_boxes = []
    for box_3d in boxes_3d:
        box_2d = project_3d_box_to_image(lidar2cam, box_3d, (image.shape[1], image.shape[0]))
        if box_2d is not None:
            lidar_boxes.append(box_2d)
    
    # --- Step 4: Get Camera Detections Bounding Boxes ---
    camera_boxes = []
    for det in detections:
        x, y, w, h = det[1]
        camera_boxes.append(np.array([x, y, x+w, y+h]))
    
    # --- Step 5: Associate Detections ---
    if len(lidar_boxes) == 0 or len(camera_boxes) == 0:
        matches = []
    else:
        matches, _, _ = fu.associate(lidar_boxes, camera_boxes)
    redundancy_count = len(matches)
    print(f"[{cam_sensor}] Redundancy Check: {redundancy_count} overlapping detections between LiDAR and camera")
    
    # --- Step 6: Visualize Results ---
    # Draw camera detections in red.
    for box in camera_boxes:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Draw LiDAR 2D boxes in green.
    for box in lidar_boxes:
        u1, v1, u2, v2 = box.astype(int)
        cv2.rectangle(image, (u1, v1), (u2, v2), (0, 255, 0), 2)
    
    # Optionally, overlay projected LiDAR points (as blue dots).
    pts_3D, pts_2D = lu.get_lidar_on_image(lidar2cam, point_cloud, (image.shape[1], image.shape[0]))
    for pt in pts_2D:
        u, v = int(pt[0]), int(pt[1])
        cv2.circle(image, (u, v), 1, (255, 0, 0), -1)
    
    fused_image = image.copy()
    
    if display_image:
        cv2.imshow(f"Fused Image - {cam_sensor}", fused_image)
        cv2.waitKey(0)
    if save_image:
        out_dir = os.path.join("output", "images")
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, f"fused_{cam_sensor}_{sample_token}.png"), fused_image)
    
    return fused_image, redundancy_count

############################################################
# Main Function
############################################################
def main():
    # --------------------------
    # Set Up nuScenes API.
    # --------------------------
    nuscenes_data_dir = ''  # Update this to nuScenes path.
    version = "v1.0-mini"  
    nusc = NuScenes(version=version, dataroot=nuscenes_data_dir, verbose=True)
    
    # --------------------------
    # Load YOLOv8 Model.
    # --------------------------
    yolo_model = YOLO("yolov8n.pt")  # Ensure the model file is available.
    
    # --------------------------
    # Select a Sample with Required Sensors.
    # --------------------------
    sample_tokens = []
    for sample in nusc.sample:
        if "CAM_FRONT" in sample["data"] and "CAM_FRONT_RIGHT" in sample["data"] and "LIDAR_TOP" in sample["data"]:
            sample_tokens.append(sample["token"])
    if not sample_tokens:
        print("No valid samples found with CAM_FRONT, CAM_FRONT_RIGHT, and LIDAR_TOP.")
        return
    
    sample_token = sample_tokens[0]
    print(f"Processing sample: {sample_token}")
    
    # --------------------------
    # Process CAM_FRONT.
    # --------------------------
    print("Processing CAM_FRONT...")
    fused_front, redundancy_front = mid_level_fusion_nuscenes(nusc, sample_token, "CAM_FRONT", "LIDAR_TOP", yolo_model,
                                                               display_image=True, save_image=False)
    
    # --------------------------
    # Process CAM_FRONT_RIGHT.
    # --------------------------
    print("Processing CAM_FRONT_RIGHT...")
    fused_front_right, redundancy_front_right = mid_level_fusion_nuscenes(nusc, sample_token, "CAM_FRONT_RIGHT", "LIDAR_TOP", yolo_model,
                                                                           display_image=True, save_image=False)
    
    print("====== Redundancy Check Results ======")
    print(f"CAM_FRONT redundant detections: {redundancy_front}")
    print(f"CAM_FRONT_RIGHT redundant detections: {redundancy_front_right}")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
