import cv2
import numpy as np
import math
import torch
from ultralytics import YOLO
import os

# Import constants
from app.skeleton_lstm import FEATURE_SIZE, SEQUENCE_LENGTH 

# Load YOLOv11n-Pose model globally
try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_paths = [
        os.path.join(project_root, 'yolo11n-pose.pt'),
        'yolo11n-pose.pt',
        os.path.join('..', 'yolo11n-pose.pt'),
    ]
    
    model_loaded = False
    for model_path in model_paths:
        if os.path.exists(model_path):
            YOLO_MODEL = YOLO(model_path)
            YOLO_MODEL.to('cpu')
            print(f"[SUCCESS] YOLOv11n-Pose model loaded from: {os.path.abspath(model_path)}")
            model_loaded = True
            break
    
    if not model_loaded:
        YOLO_MODEL = YOLO('yolo11n-pose.pt')
        YOLO_MODEL.to('cpu')
        print("[SUCCESS] YOLOv11n-Pose model loaded")
except Exception as e:
    print(f"[ERROR] YOLOv11n-Pose failed to load: {e}")
    YOLO_MODEL = None

# YOLOv11 Pose keypoint connections (17 keypoints)
YOLOV11_SKELETON = [
    [0, 1], [0, 2],  # nose to eyes
    [1, 3], [2, 4],  # eyes to ears
    [5, 6],  # shoulders
    [5, 7], [7, 9],  # left arm
    [6, 8], [8, 10],  # right arm
    [5, 11], [6, 12],  # shoulders to hips
    [11, 12],  # hips
    [11, 13], [13, 15],  # left leg
    [12, 14], [14, 16],  # right leg
]

def detect_multiple_people(image, mp_pose_instance=None, use_hog=False):
    """
    Fast detection using YOLOv11n-Pose only.
    Returns list of people with keypoints and bounding boxes.
    """
    h, w, _ = image.shape
    
    if h < 100 or w < 100 or YOLO_MODEL is None:
        return []
    
    try:
        # Single fast detection - optimized for speed
        results = YOLO_MODEL(image, conf=0.30, iou=0.50, verbose=False, half=False)
        
        people = []
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None and result.keypoints is not None:
                for person_idx in range(len(result.boxes)):
                    try:
                        box = result.boxes[person_idx]
                        kpts = result.keypoints.data[person_idx]
                        conf = float(box.conf) if hasattr(box, 'conf') else 0.5
                        
                        # Extract valid keypoints
                        valid_keypoints = []
                        for kpt in kpts:
                            if len(kpt) >= 3:
                                x, y, conf_kpt = float(kpt[0]), float(kpt[1]), float(kpt[2])
                                if conf_kpt > 0.3 and (0 <= x < w) and (0 <= y < h):
                                    valid_keypoints.append([x, y, conf_kpt])
                        
                        # Need at least 5 keypoints
                        if len(valid_keypoints) >= 5:
                            keypoint_array = np.array(valid_keypoints)
                            x_coords = keypoint_array[:, 0]
                            y_coords = keypoint_array[:, 1]
                            
                            min_x, max_x = int(np.min(x_coords)), int(np.max(x_coords))
                            min_y, max_y = int(np.min(y_coords)), int(np.max(y_coords))
                            
                            width = max_x - min_x
                            height = max_y - min_y
                            
                            if width > 20 and height > 30:
                                # Store raw keypoints for drawing
                                people.append({
                                    'keypoints': kpts.cpu().numpy() if hasattr(kpts, 'cpu') else np.array(kpts),
                                    'bbox': (min_x, min_y, width, height),
                                    'x': (min_x + max_x) / 2,
                                    'y': (min_y + max_y) / 2,
                                    'area': width * height,
                                    'confidence': conf,
                                })
                    except Exception as e:
                        continue
        
        # Sort by area
        people.sort(key=lambda p: p['area'], reverse=True)
        return people
        
    except Exception as e:
        print(f"[ERROR] YOLOv11 detection failed: {e}")
        return []

def draw_skeleton_yolo(image, keypoints, color=(0, 255, 0), thickness=2):
    """Draw skeleton directly from YOLOv11 keypoints using OpenCV"""
    if keypoints is None or len(keypoints) == 0:
        return image
    
    h, w = image.shape[:2]
    
    # Convert keypoints to numpy if needed
    if hasattr(keypoints, 'cpu'):
        kpts = keypoints.cpu().numpy()
    else:
        kpts = np.array(keypoints)
    
    # Draw connections
    for connection in YOLOV11_SKELETON:
        if len(connection) == 2:
            idx1, idx2 = connection
            if idx1 < len(kpts) and idx2 < len(kpts):
                pt1 = kpts[idx1]
                pt2 = kpts[idx2]
                
                if len(pt1) >= 3 and len(pt2) >= 3:
                    x1, y1, conf1 = float(pt1[0]), float(pt1[1]), float(pt1[2])
                    x2, y2, conf2 = float(pt2[0]), float(pt2[1]), float(pt2[2])
                    
                    if conf1 > 0.3 and conf2 > 0.3 and (0 <= x1 < w) and (0 <= y1 < h) and (0 <= x2 < w) and (0 <= y2 < h):
                        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    
    # Draw keypoints
    for kpt in kpts:
        if len(kpt) >= 3:
            x, y, conf = float(kpt[0]), float(kpt[1]), float(kpt[2])
            if conf > 0.3 and (0 <= x < w) and (0 <= y < h):
                cv2.circle(image, (int(x), int(y)), 4, color, -1)
    
    return image

def extract_8_kinematic_features(keypoints, frame_width=640, frame_height=480):
    """
    Extract features directly from YOLOv11 keypoints.
    Returns 8-feature vector: [HWR, TorsoAngle, D, P40, HipVx, H, FallAngleD, HipVy]
    """
    if keypoints is None or len(keypoints) == 0:
        return np.zeros(8, dtype=np.float32)
    
    try:
        # Convert to numpy
        if hasattr(keypoints, 'cpu'):
            kpts = keypoints.cpu().numpy()
        else:
            kpts = np.array(keypoints)
        
        # Get key body parts (YOLOv11 indices)
        # 0: nose, 5: left_shoulder, 6: right_shoulder, 11: left_hip, 12: right_hip
        nose_idx, l_shoulder_idx, r_shoulder_idx = 0, 5, 6
        l_hip_idx, r_hip_idx = 11, 12
        
        def get_point(idx):
            if idx < len(kpts) and len(kpts[idx]) >= 3:
                conf = float(kpts[idx][2])
                if conf > 0.3:
                    return float(kpts[idx][0]) / frame_width, float(kpts[idx][1]) / frame_height
            return None, None
        
        nose_x, nose_y = get_point(nose_idx)
        l_shoulder_x, l_shoulder_y = get_point(l_shoulder_idx)
        r_shoulder_x, r_shoulder_y = get_point(r_shoulder_idx)
        l_hip_x, l_hip_y = get_point(l_hip_idx)
        r_hip_x, r_hip_y = get_point(r_hip_idx)
        
        # Need at least shoulders and hips
        if l_shoulder_x is None or r_shoulder_x is None or l_hip_x is None or r_hip_x is None:
            return np.zeros(8, dtype=np.float32)
        
        # Calculate centers
        shoulder_center_x = (l_shoulder_x + r_shoulder_x) / 2
        shoulder_center_y = (l_shoulder_y + r_shoulder_y) / 2
        hip_center_x = (l_hip_x + r_hip_x) / 2
        hip_center_y = (l_hip_y + r_hip_y) / 2
        
        # Get all visible keypoints for bounding box
        visible_x = []
        visible_y = []
        for kpt in kpts:
            if len(kpt) >= 3 and float(kpt[2]) > 0.3:
                visible_x.append(float(kpt[0]) / frame_width)
                visible_y.append(float(kpt[1]) / frame_height)
        
        if len(visible_x) < 5:
            return np.zeros(8, dtype=np.float32)
        
        min_x, max_x = min(visible_x), max(visible_x)
        min_y, max_y = min(visible_y), max(visible_y)
        
        # 0. HWR (Height-to-Width Ratio)
        W = max_x - min_x
        H = max_y - min_y
        HWR = H / W if W > 0 else 0.0
        
        # 1. Torso Angle (from vertical)
        torso_x_diff = hip_center_x - shoulder_center_x
        torso_y_diff = shoulder_center_y - hip_center_y
        TorsoAngle = np.degrees(np.arctan2(abs(torso_x_diff), abs(torso_y_diff))) if torso_y_diff != 0 else 0.0
        
        # 2. D (Head to hip vertical distance)
        D = nose_y - hip_center_y if nose_y is not None else 0.0
        
        # 3-4. P40, HipVx, HipVy (velocity - set to 0)
        P40 = 0.0
        HipVx = 0.0
        HipVy = 0.0
        
        # 5. H (Hip height in frame)
        H_norm = hip_center_y
        
        # 6. Fall Angle D (Body angle from horizontal)
        FallAngleD = abs(90.0 - TorsoAngle)
        
        return np.array([HWR, TorsoAngle, D, P40, HipVx, H_norm, FallAngleD, HipVy], dtype=np.float32)
        
    except Exception as e:
        return np.zeros(8, dtype=np.float32)

def extract_55_features(image, mp_pose_instance=None):
    """Extract 55 features using YOLOv11-Pose"""
    feature_vec_55 = np.zeros(FEATURE_SIZE, dtype=np.float32)
    bbox = None
    
    try:
        people = detect_multiple_people(image, mp_pose_instance)
        
        if not people:
            return feature_vec_55, bbox
        
        # Use the largest person
        person = people[0]
        keypoints = person['keypoints']
        bbox = person['bbox']
        
        # Extract 8 kinematic features
        features_8 = extract_8_kinematic_features(keypoints, image.shape[1], image.shape[0])
        
        # Pad to 55 features
        feature_vec_55[:8] = features_8
        
        return feature_vec_55, bbox
    
    except Exception as e:
        return feature_vec_55, bbox

def predict_torch(model, input_tensor, threshold=0.75):
    """Run LSTM inference"""
    if model is None:
        return False, 0.0
    
    try:
        output = model(input_tensor)
        
        if len(output.shape) > 1:
            prob = float(output[0, 0])
        else:
            prob = float(output[0])
        
        prob = max(0.0, min(1.0, prob))
        return prob >= threshold, prob
    
    except Exception as e:
        return False, 0.0
