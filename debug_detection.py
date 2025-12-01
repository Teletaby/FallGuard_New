#!/usr/bin/env python3
"""
Debug script to diagnose fall detection and multi-person issues
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os

# Import from app
from app.video_utils import detect_multiple_people, extract_8_kinematic_features, extract_55_features

def test_detection_on_frame():
    """Test detection on a single frame from a video"""
    print("\n" + "="*70)
    print("Testing Detection on Real Frame")
    print("="*70)
    
    # Find a test video
    video_path = None
    if os.path.exists("uploads"):
        for file in os.listdir("uploads"):
            if file.endswith((".mp4", ".avi", ".mov")):
                video_path = os.path.join("uploads", file)
                break
    
    if not video_path:
        print("[WARNING] No video found in uploads folder")
        return
    
    print(f"\n[INFO] Reading video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("[ERROR] Could not open video")
        return
    
    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Could not read frame")
        return
    
    frame = cv2.resize(frame, (640, 480))
    print(f"[INFO] Frame shape: {frame.shape}")
    
    # Test detection
    print("\n[DETECTION] Running detect_multiple_people...")
    people = detect_multiple_people(frame, None, use_hog=False)
    
    print(f"[DETECTION] Found {len(people)} person(s)")
    
    if len(people) > 0:
        for idx, person in enumerate(people):
            print(f"\n  Person {idx + 1}:")
            print(f"    Bbox: {person['bbox']}")
            print(f"    Confidence: {person['confidence']:.4f}")
            print(f"    Position: ({person['x']:.1f}, {person['y']:.1f})")
            print(f"    Area: {person['area']}")
            
            # Extract features
            landmarks = person['landmarks']
            features_8 = extract_8_kinematic_features(landmarks)
            
            print(f"    Features (8):")
            print(f"      HWR: {features_8[0]:.4f}")
            print(f"      TorsoAngle: {features_8[1]:.4f}°")
            print(f"      D: {features_8[2]:.4f}")
            print(f"      P40: {features_8[3]:.4f}")
            print(f"      HipVx: {features_8[4]:.4f}")
            print(f"      H: {features_8[5]:.4f}")
            print(f"      FallAngleD: {features_8[6]:.4f}°")
            print(f"      HipVy: {features_8[7]:.4f}")
            
            # Predict fall using heuristic
            HWR = features_8[0]
            TorsoAngle = features_8[1]
            H = features_8[5]
            FallAngleD = features_8[6]
            
            fall_score = 0.0
            
            if 0.0 < HWR < 0.45:
                fall_score += 0.35
                if HWR < 0.30:
                    fall_score += 0.30
            
            if TorsoAngle > 65:
                fall_score += 0.30
                if TorsoAngle > 78:
                    fall_score += 0.15
            
            if H > 0.72:
                fall_score += 0.08
                if H > 0.82:
                    fall_score += 0.12
            
            if FallAngleD < 20:
                fall_score += 0.35
                if FallAngleD < 10:
                    fall_score += 0.18
            
            print(f"\n    Fall Score Breakdown:")
            print(f"      HWR ({HWR:.3f} < 0.45?): ", end="")
            if 0.0 < HWR < 0.45:
                print(f"+0.35, extra={0.30 if HWR < 0.30 else 0}")
            else:
                print("0")
            
            print(f"      TorsoAngle ({TorsoAngle:.1f}° > 65°?): ", end="")
            if TorsoAngle > 65:
                print(f"+0.30, extra={0.15 if TorsoAngle > 78 else 0}")
            else:
                print("0")
            
            print(f"      H ({H:.3f} > 0.72?): ", end="")
            if H > 0.72:
                print(f"+0.08, extra={0.12 if H > 0.82 else 0}")
            else:
                print("0")
            
            print(f"      FallAngleD ({FallAngleD:.1f}° < 20°?): ", end="")
            if FallAngleD < 20:
                print(f"+0.35, extra={0.18 if FallAngleD < 10 else 0}")
            else:
                print("0")
            
            threshold = 0.75
            is_fall = fall_score >= threshold
            print(f"\n    TOTAL SCORE: {fall_score:.3f}")
            print(f"    THRESHOLD: {threshold}")
            print(f"    RESULT: {'FALL' if is_fall else 'NORMAL'}")
    
    cap.release()

def test_yolo_directly():
    """Test YOLOv11 detection directly"""
    print("\n" + "="*70)
    print("Testing YOLOv11 Directly")
    print("="*70)
    
    # Load model
    model_path = 'yolo11n-pose.pt'
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        return
    
    print(f"\n[INFO] Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Create test frame
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
    
    print("[INFO] Running YOLOv11 on empty frame...")
    results = model(frame, conf=0.25, iou=0.50, verbose=False)
    
    if results:
        result = results[0]
        print(f"[DETECTION] Boxes: {len(result.boxes) if result.boxes else 0}")
        print(f"[DETECTION] Keypoints: {len(result.keypoints.data) if result.keypoints else 0}")
    
    # Try with a real video
    video_path = None
    if os.path.exists("uploads"):
        for file in os.listdir("uploads"):
            if file.endswith((".mp4", ".avi", ".mov")):
                video_path = os.path.join("uploads", file)
                break
    
    if video_path:
        print(f"\n[INFO] Testing on real video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            frame = cv2.resize(frame, (640, 480))
            print("[INFO] Running YOLOv11 on video frame...")
            results = model(frame, conf=0.25, iou=0.50, verbose=False)
            
            if results:
                result = results[0]
                num_boxes = len(result.boxes) if result.boxes else 0
                num_kpts = len(result.keypoints.data) if result.keypoints else 0
                print(f"[DETECTION] Found {num_boxes} boxes and {num_kpts} keypoint sets")

def main():
    print("\n" + "="*70)
    print("FALLGUARD DETECTION DEBUG")
    print("="*70)
    
    test_yolo_directly()
    test_detection_on_frame()
    
    print("\n" + "="*70)
    print("Debug Complete")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
