import cv2
import numpy as np
import math
import torch
import torch.nn.functional as F
from ultralytics import YOLO

# Import constants to ensure input sequence shapes match the model
from app.skeleton_lstm import FEATURE_SIZE, SEQUENCE_LENGTH 

# Load YOLOv11n-Pose model globally
try:
    YOLO_MODEL = YOLO('yolov11n-pose.pt')
    YOLO_MODEL.to('cpu')  # Ensure on CPU
    print("[SUCCESS] YOLOv11n-Pose model loaded")
except Exception as e:
    print(f"[ERROR] YOLOv11n-Pose failed to load: {e}")
    YOLO_MODEL = None

# YOLOv11 Pose keypoint mapping (17 keypoints)
YOLOV11_KEYPOINTS = {
    0: 'nose',
    1: 'left_eye', 2: 'right_eye',
    3: 'left_ear', 4: 'right_ear',
    5: 'left_shoulder', 6: 'right_shoulder',
    7: 'left_elbow', 8: 'right_elbow',
    9: 'left_wrist', 10: 'right_wrist',
    11: 'left_hip', 12: 'right_hip',
    13: 'left_knee', 14: 'right_knee',
    15: 'left_ankle', 16: 'right_ankle'
}

def detect_multiple_people(image, mp_pose_instance=None, use_hog=False):
    """
    Detects multiple people using YOLOv11-Pose exclusively.
    Returns list of people with landmarks and bounding boxes.
    
    YOLOv11 is 20-30% faster and more accurate than YOLOv8.
    Detects people at greater distances and handles occlusions better.
    """
    people = []
    h, w, _ = image.shape
    
    if h < 100 or w < 100:
        return people
    
    if YOLO_MODEL is None:
        return people
    
    try:
        # Run YOLOv11-Pose with optimized settings for accuracy and speed
        # Lower confidence (0.2) for distance detection
        # Higher IoU for better multi-person detection
        results = YOLO_MODEL(image, conf=0.2, iou=0.5, verbose=False)
        
        if results and len(results) > 0:
            result = results[0]
            
            # Extract bounding boxes and keypoints
            if result.boxes is not None and result.keypoints is not None:
                num_people = len(result.boxes)
                
                for person_idx in range(num_people):
                    try:
                        box = result.boxes[person_idx]
                        kpts = result.keypoints.data[person_idx]
                        conf = float(box.conf) if hasattr(box, 'conf') else 0.5
                        
                        # Extract valid keypoints (confidence > 0.3)
                        valid_keypoints = []
                        keypoint_confidences = []
                        
                        for kpt_idx, kpt in enumerate(kpts):
                            if len(kpt) >= 3:
                                x, y, conf_kpt = float(kpt[0]), float(kpt[1]), float(kpt[2])
                                # Increased threshold for more reliable keypoints
                                if conf_kpt > 0.5 and (0 <= x < w) and (0 <= y < h):
                                    valid_keypoints.append([x, y, conf_kpt])
                                    keypoint_confidences.append(conf_kpt)
                        
                        # Need at least 5 valid keypoints for a person (down from 8 for distance)
                        if len(valid_keypoints) >= 5:
                            # Calculate bounding box from keypoints
                            keypoint_array = np.array(valid_keypoints)
                            x_coords = keypoint_array[:, 0]
                            y_coords = keypoint_array[:, 1]
                            
                            min_x, max_x = np.min(x_coords), np.max(x_coords)
                            min_y, max_y = np.min(y_coords), np.max(y_coords)
                            
                            width = max_x - min_x
                            height = max_y - min_y
                            
                            # Minimum size: 8x12 pixels (very small, distant people)
                            if width > 8 and height > 12:
                                # Add padding for bounding box
                                pad_x = width * 0.15
                                pad_y = height * 0.15
                                
                                bx = max(0, int(min_x - pad_x))
                                by = max(0, int(min_y - pad_y))
                                bw = min(w - bx, int(width + 2 * pad_x))
                                bh = min(h - by, int(height + 2 * pad_y))
                                
                                # Convert YOLOv11 keypoints to MediaPipe-compatible format
                                # YOLOv11 has 17 keypoints, MediaPipe format has 33
                                landmarks = create_mediapipe_landmarks_from_yolov11(kpts, w, h)
                                
                                # Calculate average confidence from valid keypoints
                                avg_confidence = float(np.mean(keypoint_confidences)) if keypoint_confidences else conf
                                
                                people.append({
                                    'landmarks': landmarks,
                                    'bbox': (bx, by, bw, bh),
                                    'x': bx + bw / 2,
                                    'y': by + bh / 2,
                                    'area': bw * bh,
                                    'confidence': avg_confidence,
                                    'keypoints_raw': kpts,  # Keep raw keypoints for accuracy
                                    'person_idx': person_idx
                                })
                    
                    except Exception as e:
                        continue
        
        # Sort by area (largest people first)
        people.sort(key=lambda p: p['area'], reverse=True)
        
    except Exception as e:
        print(f"[ERROR] YOLOv11-Pose detection failed: {e}")
    
    return people

def create_mediapipe_landmarks_from_yolov11(keypoints, frame_width, frame_height):
    """
    Converts YOLOv11 keypoints (17 points) to MediaPipe format (33 landmarks).
    This ensures compatibility with existing feature extraction code.
    """
    import mediapipe as mp
    
    landmarks = mp.solutions.pose.PoseLandmarkList()
    
    # YOLOv11 to MediaPipe keypoint mapping
    yolo_to_mediapipe = {
        0: 0,   # nose -> nose
        1: 2,   # left_eye -> left_eye
        2: 5,   # right_eye -> right_eye
        3: 3,   # left_ear -> left_ear_unknown (use 3)
        4: 6,   # right_ear -> right_ear_unknown (use 6)
        5: 11,  # left_shoulder -> left_shoulder
        6: 12,  # right_shoulder -> right_shoulder
        7: 13,  # left_elbow -> left_elbow
        8: 14,  # right_elbow -> right_elbow
        9: 15,  # left_wrist -> left_wrist
        10: 16, # right_wrist -> right_wrist
        11: 23, # left_hip -> left_hip
        12: 24, # right_hip -> right_hip
        13: 25, # left_knee -> left_knee
        14: 26, # right_knee -> right_knee
        15: 27, # left_ankle -> left_ankle
        16: 28  # right_ankle -> right_ankle
    }
    
    # Create 33 landmarks with zeros
    for i in range(33):
        lm = mp.solutions.pose.PoseLandmark(x=0, y=0, z=0, visibility=0)
        landmarks.landmark.append(lm)
    
    # Fill in the detected keypoints
    for yolo_idx, kpt in enumerate(keypoints):
        if yolo_idx in yolo_to_mediapipe:
            mediapipe_idx = yolo_to_mediapipe[yolo_idx]
            if len(kpt) >= 3:
                x = float(kpt[0]) / frame_width
                y = float(kpt[1]) / frame_height
                conf = float(kpt[2])
                
                # Clamp to valid range
                x = max(0, min(1, x))
                y = max(0, min(1, y))
                
                lm = mp.solutions.pose.PoseLandmark(x=x, y=y, z=0, visibility=conf)
                landmarks.landmark[mediapipe_idx] = lm
    
    return landmarks

def draw_skeleton(image, landmarks, is_fall_confirmed):
    """
    Draws the skeleton and keypoints on the image.
    Works with both YOLOv11 keypoints and MediaPipe landmarks.
    """
    import mediapipe as mp
    
    if landmarks:
        mp_drawing = mp.solutions.drawing_utils
        
        # Change color to red if fall confirmed, else blue
        line_color = (0, 0, 255) if is_fall_confirmed else (255, 0, 0)
        point_color = (0, 255, 255) if is_fall_confirmed else (0, 255, 0)
        
        try:
            mp_drawing.draw_landmarks(
                image,
                landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=point_color, thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=line_color, thickness=2)
            )
        except Exception as e:
            pass  # Silently fail if drawing fails
    
    return image

# --- Feature Extraction Helpers ---
def calculate_distance(p1, p2):
    """Calculates Euclidean distance between two points."""
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def extract_8_kinematic_features(landmarks):
    """
    Calculates 8 kinematic features for fall detection.
    Works with MediaPipe format landmarks (from YOLOv11).
    """
    import mediapipe as mp
    
    if not landmarks or not landmarks.landmark:
        return np.zeros(8, dtype=np.float32)
    
    mp_pose = mp.solutions.pose
    
    try:
        # Get key body parts
        L_SHOULDER = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        R_SHOULDER = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        L_HIP = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value]
        R_HIP = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value]
        NOSE = landmarks.landmark[mp_pose.PoseLandmark.NOSE.value]
        
        # Get bounding box from visible landmarks
        x_coords = [lm.x for lm in landmarks.landmark if lm.visibility > 0.3]
        y_coords = [lm.y for lm in landmarks.landmark if lm.visibility > 0.3]
        
        if not x_coords or not y_coords:
            return np.zeros(8, dtype=np.float32)
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # --- 0. HWR (Height-to-Width Ratio) ---
        W = max_x - min_x
        H = max_y - min_y
        HWR = H / W if W > 0 else 0.0
        
        # --- 1. Torso Angle (Angle from vertical) ---
        shoulder_center_x = (L_SHOULDER.x + R_SHOULDER.x) / 2
        shoulder_center_y = (L_SHOULDER.y + R_SHOULDER.y) / 2
        hip_center_x = (L_HIP.x + R_HIP.x) / 2
        hip_center_y = (L_HIP.y + R_HIP.y) / 2
        
        torso_x_diff = hip_center_x - shoulder_center_x
        torso_y_diff = shoulder_center_y - hip_center_y
        
        TorsoAngle = np.degrees(np.arctan2(abs(torso_x_diff), abs(torso_y_diff))) if torso_y_diff != 0 else 0.0
        
        # --- 2. D (Head to hip vertical distance) ---
        D = NOSE.y - hip_center_y
        
        # --- 3-4. P40, HipVx, HipVy (velocity features - set to 0) ---
        P40 = 0.0
        HipVx = 0.0
        HipVy = 0.0
        
        # --- 5. H (Hip height in frame) ---
        H_norm = hip_center_y
        
        # --- 6. Fall Angle D (Body angle from horizontal) ---
        FallAngleD = abs(90.0 - TorsoAngle)
        
        # Create 8-feature vector
        features_8 = np.array([HWR, TorsoAngle, D, P40, HipVx, H_norm, FallAngleD, HipVy], dtype=np.float32)
        
        return features_8
    
    except Exception as e:
        return np.zeros(8, dtype=np.float32)

def extract_55_features(image, mp_pose_instance=None):
    """
    Extracts 55 features using YOLOv11-Pose.
    Returns feature vector for LSTM or heuristic classification.
    """
    # Process all detected people, not just the largest
    people_features = []
    try:
        people = detect_multiple_people(image, mp_pose_instance)
        if not people:
            return people_features  # Empty list if no people
        for person in people:
            feature_vec_55 = np.zeros(FEATURE_SIZE, dtype=np.float32)
            landmarks = person['landmarks']
            bbox = person['bbox']
            features_8 = extract_8_kinematic_features(landmarks)
            feature_vec_55[:8] = features_8
            people_features.append({
                'features': feature_vec_55,
                'bbox': bbox,
                'landmarks': landmarks,
                'person_idx': person.get('person_idx', None)
            })
        return people_features
    except Exception as e:
        return people_features

# New helper: run fall detection for all people in a frame
def detect_falls_for_all_people(model, image, mp_pose_instance=None, threshold=0.75):
    """
    Runs fall detection for all detected people in the image.
    Returns a list of dicts: {is_fall, confidence, bbox, person_idx}
    """
    results = []
    people_features = extract_55_features(image, mp_pose_instance)
    for person in people_features:
        features = person['features']
        bbox = person['bbox']
        person_idx = person['person_idx']
        # Prepare input for model
        input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        is_fall, confidence = predict_torch(model, input_tensor, threshold=threshold)
        results.append({
            'is_fall': is_fall,
            'confidence': confidence,
            'bbox': bbox,
            'person_idx': person_idx
        })
    return results

def predict_torch(model, input_tensor, threshold=0.75):
    """
    Runs LSTM inference on feature tensor.
    Returns (is_fall, confidence).
    """
    if model is None:
        return False, 0.0
    
    try:
        output = model(input_tensor)
        
        # Handle different output shapes
        if len(output.shape) > 1:
            prob = float(output[0, 0])
        else:
            prob = float(output[0])
        
        # Clamp probability to [0, 1]
        prob = max(0.0, min(1.0, prob))
        
        return prob >= threshold, prob
    
    except Exception as e:
        return False, 0.0
