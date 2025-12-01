#!/usr/bin/env python3
"""
Test with LSTM disabled
"""

import cv2
import numpy as np
import os

def test_without_lstm():
    """Test detection with LSTM disabled"""
    
    video_path = "uploads/12.mp4"
    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        return
    
    print(f"\n[TEST] Testing with LSTM DISABLED")
    print("="*70)
    
    from app.video_utils import detect_multiple_people, extract_8_kinematic_features
    from collections import deque
    
    cap = cv2.VideoCapture(video_path)
    
    sequence = deque([np.zeros(55, dtype=np.float32) for _ in range(30)], maxlen=30)
    consecutive_fall_frames = 0
    frame_count = 0
    
    GLOBAL_SETTINGS = {"fall_threshold": 0.75}
    
    print("\n[INFO] Processing 50 frames with HEURISTIC ONLY (LSTM disabled)...\n")
    
    while frame_count < 50:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (640, 480))
        frame_count += 1
        
        people = detect_multiple_people(frame, None, use_hog=False)
        
        if not people:
            if frame_count % 5 == 0:
                print(f"Frame {frame_count:3d}: No detection")
            consecutive_fall_frames = 0
            continue
        
        person = people[0]
        landmarks = person['landmarks']
        features_8 = extract_8_kinematic_features(landmarks)
        feature_vec = np.zeros(55, dtype=np.float32)
        feature_vec[:8] = features_8
        sequence.append(feature_vec)
        
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
        
        is_falling = fall_score >= GLOBAL_SETTINGS['fall_threshold']
        
        if is_falling:
            consecutive_fall_frames += 1
        else:
            consecutive_fall_frames = 0
        
        fall_confirmed = consecutive_fall_frames >= 7
        
        status = "FALL CONFIRMED!!!" if fall_confirmed else ("fall detected" if is_falling else "NORMAL")
        
        if frame_count % 3 == 0 or is_falling or fall_confirmed:
            print(f"Frame {frame_count:3d}: HWR={HWR:.2f}, Torso={TorsoAngle:5.1f}°, H={H:.2f}, Fall°={FallAngleD:5.1f}°")
            print(f"           Score={fall_score:.2f} → {status}")
            if is_falling or fall_confirmed:
                print()
    
    cap.release()
    
    print("\n" + "="*70)
    print(f"[RESULT] Maximum consecutive fall frames: {consecutive_fall_frames}/7")
    if consecutive_fall_frames >= 7:
        print("[RESULT] FALL WAS DETECTED")
    else:
        print(f"[RESULT] NO FALL DETECTED - Person remained in NORMAL pose")

if __name__ == "__main__":
    try:
        test_without_lstm()
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
