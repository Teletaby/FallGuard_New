import cv2
import numpy as np
import math
import torch
import torch.nn.functional as F
import mediapipe as mp 
from ultralytics import YOLO

# Import constants to ensure input sequence shapes match the model
from app.skeleton_lstm import FEATURE_SIZE, SEQUENCE_LENGTH 

# Load YOLOv8-Pose model globally (has skeleton detection built-in)
try:
    YOLO_MODEL = YOLO('yolov8n-pose.pt')  # Nano pose model for speed + skeleton
    print("[SUCCESS] YOLOv8-Pose model loaded")
except Exception as e:
    print(f"[WARNING] YOLOv8-Pose not available: {e}. Will use MediaPipe only.")
    YOLO_MODEL = None

# --- Multi-Person Detection Support ---
def detect_multiple_people(image, mp_pose_instance, use_hog=False):
    """
    Uses YOLOv8-Pose for skeleton detection (has keypoints built-in).
    Falls back to MediaPipe if YOLOv8 unavailable.
    """
    people = []
    h, w, _ = image.shape
    
    if h < 100 or w < 100:
        return people
    
    try:
        # PRIMARY: YOLOv8-Pose (detects people + skeleton keypoints together)
        if YOLO_MODEL is not None:
            try:
                # Use lower confidence (0.25) to detect people at distance
                results_yolo = YOLO_MODEL(image, conf=0.25, verbose=False)
                
                for result in results_yolo:
                    # Check if boxes exist and have elements
                    num_boxes = len(result.boxes) if (result.boxes is not None and hasattr(result.boxes, '__len__')) else 0
                    
                    if num_boxes == 0:
                        continue
                    
                    if result.keypoints is not None:
                        for box_idx, box in enumerate(result.boxes):
                            cls_val = int(box.cls) if hasattr(box, 'cls') else -1
                            
                            if cls_val == 0:  # Person class
                                # Get keypoints for this person
                                kpts = result.keypoints.data[box_idx]
                                
                                # Filter valid keypoints (not all zeros)
                                valid_kpts = []
                                for kpt in kpts:
                                    if len(kpt) >= 2 and (kpt[0] > 0 or kpt[1] > 0):
                                        valid_kpts.append(kpt)
                                
                                if len(valid_kpts) >= 5:  # Reduced from 8 to detect people at distance
                                    x_coords = [float(kpt[0]) for kpt in valid_kpts]
                                    y_coords = [float(kpt[1]) for kpt in valid_kpts]
                                    
                                    min_x, max_x = min(x_coords), max(x_coords)
                                    min_y, max_y = min(y_coords), max(y_coords)
                                    
                                    width = max_x - min_x
                                    height = max_y - min_y
                                    
                                    if width > 10 and height > 15:  # Reduced minimum size for distance detection
                                        pad_x = width * 0.1
                                        pad_y = height * 0.1
                                        
                                        bx = max(0, int(min_x - pad_x))
                                        by = max(0, int(min_y - pad_y))
                                        bw = min(w - bx, int(width + 2 * pad_x))
                                        bh = min(h - by, int(height + 2 * pad_y))
                                        
                                        # Convert YOLO keypoints to MediaPipe format (17 keypoints from YOLO-Pose)
                                        landmarks = mp.solutions.pose.PoseLandmarkList()
                                        for kpt in kpts:
                                            if len(kpt) >= 2:
                                                x = float(kpt[0]) / w
                                                y = float(kpt[1]) / h
                                                conf = float(kpt[2]) if len(kpt) > 2 else 0.9
                                                x = max(0, min(1, x))
                                                y = max(0, min(1, y))
                                                lm = mp.solutions.pose.PoseLandmark(
                                                    x=x, y=y, z=0, visibility=conf
                                                )
                                                landmarks.landmark.append(lm)
                                        
                                        # Pad to 33 landmarks (MediaPipe standard)
                                        while len(landmarks.landmark) < 33:
                                            lm = mp.solutions.pose.PoseLandmark(
                                                x=0, y=0, z=0, visibility=0
                                            )
                                            landmarks.landmark.append(lm)
                                        
                                        conf = 0.9
                                        people.append({
                                            'landmarks': landmarks,
                                            'bbox': (bx, by, bw, bh),
                                            'x': bx + bw / 2,
                                            'y': by + bh / 2,
                                            'area': bw * bh,
                                            'confidence': conf
                                        })
                
                if len(people) > 0:
                    people.sort(key=lambda p: p['area'], reverse=True)
                    return people
                    
            except Exception as e:
                print(f"[ERROR] YOLOv8-Pose failed: {e}")
                import traceback
                traceback.print_exc()
        
        # FALLBACK: MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = mp_pose_instance.process(image_rgb)
        image.flags.writeable = True
        
        if results and results.pose_landmarks:
            visible_landmarks = [lm for lm in results.pose_landmarks.landmark if lm.visibility > 0.1]
            
            if len(visible_landmarks) >= 8:
                x_coords = [lm.x * w for lm in visible_landmarks]
                y_coords = [lm.y * h for lm in visible_landmarks]
                
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)
                
                width = max_x - min_x
                height = max_y - min_y
                
                pad_x = width * 0.1 if width > 0 else w * 0.02
                pad_y = height * 0.1 if height > 0 else h * 0.02
                
                bx = max(0, int(min_x - pad_x))
                by = max(0, int(min_y - pad_y))
                bw = min(w - bx, int(width + 2 * pad_x))
                bh = min(h - by, int(height + 2 * pad_y))
                
                if bw > 15 and bh > 25:
                    conf = float(np.mean([lm.visibility for lm in visible_landmarks]))
                    people.append({
                        'landmarks': results.pose_landmarks,
                        'bbox': (bx, by, bw, bh),
                        'x': bx + bw / 2,
                        'y': by + bh / 2,
                        'area': bw * bh,
                        'confidence': conf
                    })
                    print(f"[DETECTION] MediaPipe: 1 person detected")
                    print(f"[DETECTION] MediaPipe: 1 person detected")
        
        people.sort(key=lambda p: p['area'], reverse=True)
        
        # FALLBACK: If MediaPipe found nobody, try YOLOv8-Pose with lower confidence
        if len(people) == 0 and YOLO_MODEL is not None:
            try:
                results_yolo = YOLO_MODEL(image, conf=0.25, verbose=False)
                
                for result in results_yolo:
                    if result.keypoints is not None and result.boxes is not None:
                        num_boxes = len(result.boxes) if hasattr(result.boxes, '__len__') else 0
                        
                        if num_boxes > 0:
                            for box_idx, box in enumerate(result.boxes):
                                cls_val = int(box.cls) if hasattr(box, 'cls') else -1
                                
                                if cls_val == 0:  # Person class
                                    kpts = result.keypoints.data[box_idx]
                                    
                                    # Filter valid keypoints
                                    valid_kpts = []
                                    for kpt in kpts:
                                        if len(kpt) >= 2 and (kpt[0] > 0 or kpt[1] > 0):
                                            valid_kpts.append(kpt)
                                    
                                    if len(valid_kpts) >= 8:
                                        x_coords = [float(kpt[0]) for kpt in valid_kpts]
                                        y_coords = [float(kpt[1]) for kpt in valid_kpts]
                                        
                                        min_x, max_x = min(x_coords), max(x_coords)
                                        min_y, max_y = min(y_coords), max(y_coords)
                                        
                                        width = max_x - min_x
                                        height = max_y - min_y
                                        
                                        if width > 20 and height > 30:
                                            pad_x = width * 0.1
                                            pad_y = height * 0.1
                                            
                                            bx = max(0, int(min_x - pad_x))
                                            by = max(0, int(min_y - pad_y))
                                            bw = min(w - bx, int(width + 2 * pad_x))
                                            bh = min(h - by, int(height + 2 * pad_y))
                                            
                                            # Convert YOLO keypoints to MediaPipe format
                                            landmarks = mp.solutions.pose.PoseLandmarkList()
                                            for kpt in kpts:
                                                if len(kpt) >= 2:
                                                    x = float(kpt[0]) / w
                                                    y = float(kpt[1]) / h
                                                    conf = float(kpt[2]) if len(kpt) > 2 else 0.9
                                                    x = max(0, min(1, x))
                                                    y = max(0, min(1, y))
                                                    lm = mp.solutions.pose.PoseLandmark(
                                                        x=x, y=y, z=0, visibility=conf
                                                    )
                                                    landmarks.landmark.append(lm)
                                            
                                            # Pad to 33 landmarks
                                            while len(landmarks.landmark) < 33:
                                                lm = mp.solutions.pose.PoseLandmark(
                                                    x=0, y=0, z=0, visibility=0
                                                )
                                                landmarks.landmark.append(lm)
                                            
                                            people.append({
                                                'landmarks': landmarks,
                                                'bbox': (bx, by, bw, bh),
                                                'x': bx + bw / 2,
                                                'y': by + bh / 2,
                                                'area': bw * bh,
                                                'confidence': 0.9
                                            })
                
                if len(people) > 0:
                    people.sort(key=lambda p: p['area'], reverse=True)
                
            except Exception as e:
                print(f"[ERROR] YOLOv8-Pose fallback failed: {e}")
        
    except Exception as e:
        print(f"[ERROR] Detection failed: {e}")
        import traceback
        traceback.print_exc()

    return people

# --- Visualization ---
def draw_skeleton(image, results, is_fall_confirmed):
    """Draws the MediaPipe skeleton and connections on the image."""
    
    if results.pose_landmarks:
        mp_drawing = mp.solutions.drawing_utils
        
        # Change color to red if fall confirmed
        line_color = (0, 0, 255) if is_fall_confirmed else (255, 0, 0)
        point_color = (0, 255, 255) if is_fall_confirmed else (0, 255, 0)

        # Draw the pose landmarks and connections
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=point_color, thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=line_color, thickness=2)
        )
    return image


# --- Feature Extraction Helpers ---
def calculate_distance(p1, p2):
    """Calculates the Euclidean distance between two points."""
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def extract_8_kinematic_features(landmarks):
    """
    Calculates the 8 kinematic features (HWR, TorsoAngle, D, H, FallAngleD, 
    and mock values for P40, HipVx, HipVy).
    """
    mp_pose = mp.solutions.pose
    
    if not landmarks or not landmarks.landmark:
        # Return a zero vector of 8 elements if no landmarks are found
        return np.zeros(8, dtype=np.float32)

    # MediaPipe Landmark Points
    L_SHOULDER = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    R_SHOULDER = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    L_HIP = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value]
    R_HIP = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value]
    NOSE = landmarks.landmark[mp_pose.PoseLandmark.NOSE.value]

    # --- 0. HWR (Height-to-Width Ratio of Bounding Box) ---
    x_coords = [lm.x for lm in landmarks.landmark if lm.visibility > 0.5]
    y_coords = [lm.y for lm in landmarks.landmark if lm.visibility > 0.5]
    
    if not x_coords or not y_coords:
        return np.zeros(8, dtype=np.float32)

    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    W = max_x - min_x
    H = max_y - min_y
    HWR = H / W if W > 0 else 0.0

    # --- 1. Torso Angle (Angle of the main body axis to the vertical) ---
    shoulder_center_x = (L_SHOULDER.x + R_SHOULDER.x) / 2
    shoulder_center_y = (L_SHOULDER.y + R_SHOULDER.y) / 2
    hip_center_x = (L_HIP.x + R_HIP.x) / 2
    hip_center_y = (L_HIP.y + R_HIP.y) / 2
    
    torso_x_diff = hip_center_x - shoulder_center_x
    torso_y_diff = shoulder_center_y - hip_center_y # Vertical difference
    
    # Angle relative to vertical axis (0 degrees is perfectly upright)
    TorsoAngle = np.degrees(np.arctan2(abs(torso_x_diff), abs(torso_y_diff))) if torso_y_diff != 0 else 0.0

    # --- 2. D (Difference in y-coordinates of head and hip centers) ---
    D = NOSE.y - hip_center_y

    # --- 3. P40 (Average joint velocity) & 4. Hip Vx & 7. Hip Vy --- 
    # These velocity/sequence features are set to 0.0 here as they require sequential data
    P40 = 0.0 
    HipVx = 0.0 
    HipVy = 0.0 
    
    # --- 5. Height of Hip Center (H) ---
    # Normalized height relative to the frame (0.0 is ceiling, 1.0 is floor)
    H = hip_center_y 

    # --- 6. Fall Angle D (Angle of body to horizontal, 90 degrees is vertical) ---
    # Using the same angle calculation as Torso Angle, but relative to horizontal
    FallAngleD = abs(90.0 - TorsoAngle)

    # Create the 8-feature vector
    features_8 = np.array([HWR, TorsoAngle, D, P40, HipVx, H, FallAngleD, HipVy], dtype=np.float32)

    return features_8

def extract_55_features(image, mp_pose_instance):
    """
    Runs MediaPipe, calculates the 8 kinematic features, and pads the result to 55 features.
    """
    mp_pose = mp.solutions.pose
    
    image.flags.writeable = False
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = mp_pose_instance.process(image_rgb)
    image.flags.writeable = True
    
    bbox = None
    feature_vec_55 = np.zeros(FEATURE_SIZE, dtype=np.float32)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks
        
        # 1. Calculate the 8 kinematic features
        features_8 = extract_8_kinematic_features(landmarks)
        
        # 2. Pad to 55 features (to match the model's expected input size)
        feature_vec_55[:8] = features_8
        
        # 3. Calculate Bounding Box (for visualization)
        h, w, _ = image.shape
        x_coords = [lm.x * w for lm in landmarks.landmark if lm.visibility > 0.5]
        y_coords = [lm.y * h for lm in landmarks.landmark if lm.visibility > 0.5]
        
        if x_coords and y_coords:
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            x = int(min_x)
            y = int(min_y)
            bw = int(max_x - min_x)
            bh = int(max_y - min_y)
            bbox = (x, y, bw, bh)
        
    return bbox, feature_vec_55, results

# --- Prediction Utility (CRITICAL FIX APPLIED HERE) ---
def predict_torch(model, sequence_tensor, threshold=0.5):
    """
    Runs inference on the PyTorch model for a single sequence.
    
    Args:
        model (LSTMModel): The loaded PyTorch model.
        sequence_tensor (torch.Tensor): Input tensor of shape (1, SEQUENCE_LENGTH, FEATURE_SIZE).
        threshold (float): Probability threshold for fall classification.
        
    Returns:
        tuple: (prediction_class, probability_of_fall)
    """
    if model is None:
        return 0, 0.0
        
    try:
        with torch.no_grad():
            # Get the raw logits from the model
            logits = model(sequence_tensor)
            
            # Handle different output shapes
            if logits.dim() == 1:
                # Single output (shape: [1])
                prob_fall = torch.sigmoid(logits[0]).item()
                prediction = 1 if prob_fall >= threshold else 0
                return prediction, prob_fall
            elif logits.shape[1] == 1:
                # Single output in batch (shape: [1, 1])
                prob_fall = torch.sigmoid(logits[0, 0]).item()
                prediction = 1 if prob_fall >= threshold else 0
                return prediction, prob_fall
            elif logits.shape[1] == 2:
                # Two class output (shape: [1, 2])
                probs = F.softmax(logits, dim=1)
                prob_fall = probs[0, 1].item()
                prediction = 1 if prob_fall >= threshold else 0
                return prediction, prob_fall
            else:
                print(f"[WARNING] Unexpected LSTM output shape: {logits.shape}")
                return 0, 0.0
            
    except Exception as e:
        print(f"[ERROR] LSTM prediction error: {e}")
        import traceback
        traceback.print_exc()
        return 0, 0.0