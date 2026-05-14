#!/usr/bin/env python3
"""
Comprehensive diagnostic to understand fall detection and multi-person issues
"""

import cv2
import numpy as np
import os
from collections import deque

def diagnose_issues():
    """Run full diagnostics"""
    
    video_path = "uploads/12.mp4"
    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        return
    
    print("\n" + "="*80)
    print("FALLGUARD COMPREHENSIVE DIAGNOSTICS")
    print("="*80)
    
    from app.video_utils import detect_multiple_people, extract_8_kinematic_features
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n[VIDEO] Total frames: {total_frames}")
    print(f"[VIDEO] Testing first 100 frames\n")
    
    # Track detection statistics
    detection_stats = {
        'total_frames_processed': 0,
        'frames_with_detection': 0,
        'frames_no_detection': 0,
        'people_detected': [],
        'fall_detection_attempts': 0,
        'false_fall_alerts': 0,
        'true_falls': 0,
    }
    
    person_fall_states = {}
    frame_count = 0
    
    GLOBAL_SETTINGS = {"fall_threshold": 0.75}
    
    print("[DIAGNOSIS] Checking detection and fall classification...\n")
    
    while frame_count < 100:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (640, 480))
        frame_count += 1
        detection_stats['total_frames_processed'] += 1
        
        # Detect people
        people = detect_multiple_people(frame, None, use_hog=False)
        
        if not people:
            detection_stats['frames_no_detection'] += 1
            # Reset fall states when no detection
            for person_id in list(person_fall_states.keys()):
                person_fall_states[person_id]['frames'] = 0
            continue
        
        detection_stats['frames_with_detection'] += 1
        detection_stats['people_detected'].append(len(people))
        
        # Process each person
        for person_idx, person in enumerate(people):
            person_id = person_idx + 1  # Simple ID
            
            if person_id not in person_fall_states:
                person_fall_states[person_id] = {'frames': 0}
            
            landmarks = person['landmarks']
            features_8 = extract_8_kinematic_features(landmarks)
            
            if np.any(features_8 != 0):  # Valid features
                HWR = features_8[0]
                TorsoAngle = features_8[1]
                H = features_8[5]
                FallAngleD = features_8[6]
                
                # Calculate fall score
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
                detection_stats['fall_detection_attempts'] += 1
                
                if is_falling:
                    person_fall_states[person_id]['frames'] += 1
                else:
                    person_fall_states[person_id]['frames'] = 0
                
                fall_confirmed = person_fall_states[person_id]['frames'] >= 7
                
                # Log interesting frames
                if frame_count % 10 == 0 or is_falling or fall_confirmed:
                    print(f"Frame {frame_count:3d} Person {person_id}: HWR={HWR:.2f}, Torso={TorsoAngle:5.1f}°, "
                          f"H={H:.2f}, Fall°={FallAngleD:5.1f}°")
                    print(f"         Score={fall_score:.2f} (falling={is_falling}), "
                          f"Frames={person_fall_states[person_id]['frames']}/7 → "
                          f"{'FALL ALERT' if fall_confirmed else ('detecting...' if is_falling else 'NORMAL')}")
                    if fall_confirmed:
                        detection_stats['true_falls'] += 1
                        print()
    
    cap.release()
    
    # Print statistics
    print("\n" + "="*80)
    print("DIAGNOSTIC RESULTS")
    print("="*80)
    
    print(f"\n[DETECTION STATISTICS]")
    print(f"  Total frames processed: {detection_stats['total_frames_processed']}")
    print(f"  Frames with detection: {detection_stats['frames_with_detection']}")
    print(f"  Frames NO detection: {detection_stats['frames_no_detection']}")
    print(f"  Detection rate: {detection_stats['frames_with_detection']/detection_stats['total_frames_processed']*100:.1f}%")
    
    if detection_stats['people_detected']:
        avg_people = np.mean(detection_stats['people_detected'])
        max_people = max(detection_stats['people_detected'])
        print(f"  Average people per frame: {avg_people:.2f}")
        print(f"  Max people detected: {max_people}")
    
    print(f"\n[FALL DETECTION STATISTICS]")
    print(f"  Fall detection attempts: {detection_stats['fall_detection_attempts']}")
    print(f"  Confirmed falls: {detection_stats['true_falls']}")
    print(f"  False alerts: {detection_stats['false_fall_alerts']}")
    
    print(f"\n[ISSUES IDENTIFIED]")
    if detection_stats['frames_no_detection'] > detection_stats['frames_with_detection']:
        print("  ⚠ High NO DETECTION rate - YOLOv11 not detecting people")
    elif detection_stats['frames_no_detection'] > 0:
        print(f"  ⚠ Some frames ({detection_stats['frames_no_detection']}) have no detection")
    else:
        print("  ✓ Detection is reliable (100% of frames)")
    
    if max(detection_stats['people_detected']) if detection_stats['people_detected'] else 0 < 2:
        print("  ⚠ Only 1 person detected at most - need to check if video has 2+ people")
    else:
        print(f"  ✓ Multi-person detection working (max {max(detection_stats['people_detected'])} people)")
    
    if detection_stats['fall_detection_attempts'] == 0:
        print("  ⚠ No valid features extracted - skeleton may be inaccurate")
    elif detection_stats['true_falls'] == 0:
        print("  ⚠ No falls confirmed - either video has no falls or thresholds too strict")
    else:
        print(f"  ✓ Fall detection working ({detection_stats['true_falls']} falls detected)")

if __name__ == "__main__":
    try:
        diagnose_issues()
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
