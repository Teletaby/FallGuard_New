#!/usr/bin/env python3
"""
Test script to debug multi-person detection and fall detection accuracy.
This will help identify why:
1. Only 1 person is detected instead of 2
2. Fall is detected when people are just standing
"""

import cv2
import numpy as np
import os
from app.video_utils import detect_multiple_people, extract_8_kinematic_features

# Test video path
video_path = os.path.join('uploads', '586837864_25303762689290199_4978210224702831960_n.mp4')

print(f"Testing video: {video_path}")
print(f"Video exists: {os.path.exists(video_path)}")

if not os.path.exists(video_path):
    print("ERROR: Video file not found!")
    exit(1)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("ERROR: Cannot open video!")
    exit(1)

print(f"Video properties:")
print(f"  FPS: {cap.get(cv2.CAP_PROP_FPS)}")
print(f"  Frame count: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
print(f"  Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))} x {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

frame_count = 0
max_frames_to_test = 450  # Test entire video

print("\n" + "="*80)
print("SCANNING VIDEO FOR PEOPLE DETECTION...")
print("="*80)

frame_count = 0
total_frames_with_1_person = 0
total_frames_with_2_people = 0
total_frames_with_fall = 0
frames_with_detection = []

try:
    while frame_count < max_frames_to_test:
        ret, frame = cap.read()
        if not ret:
            print(f"\nEnd of video at frame {frame_count}")
            break
        
        frame = cv2.resize(frame, (640, 480))
        
        # Detect people
        people = detect_multiple_people(frame, None, use_hog=False)
        
        if people:
            frames_with_detection.append((frame_count, len(people)))
            
            if len(people) == 1:
                total_frames_with_1_person += 1
            elif len(people) >= 2:
                total_frames_with_2_people += 1
            
            # Check for fall in each person
            for person_idx, person in enumerate(people):
                landmarks = person['landmarks']
                features_8 = extract_8_kinematic_features(landmarks)
                
                # Calculate heuristic fall score
                HWR = features_8[0]        
                TorsoAngle = features_8[1] 
                D = features_8[2]          
                H = features_8[5]          
                FallAngleD = features_8[6] 
                
                fall_score = 0.0
                reasons = []
                
                # Check fall indicators
                if 0.0 < HWR < 0.45:
                    fall_score += 0.25
                    if HWR < 0.30:
                        fall_score += 0.20
                    reasons.append(f"HWR={HWR:.3f} (low)")
                
                if TorsoAngle > 70:
                    fall_score += 0.30
                    if TorsoAngle > 80:
                        fall_score += 0.15
                    reasons.append(f"TorsoAngle={TorsoAngle:.1f}° (high)")
                
                if H > 0.75:
                    fall_score += 0.15
                    reasons.append(f"H={H:.3f} (very low)")
                
                if FallAngleD < 25:
                    fall_score += 0.25
                    if FallAngleD < 15:
                        fall_score += 0.15
                    reasons.append(f"FallAngleD={FallAngleD:.1f}° (low)")
                
                if abs(D) < 0.10:
                    fall_score += 0.05
                    reasons.append(f"D={D:.3f} (compressed)")
                
                # Penalize weak cases
                if fall_score < 0.5:
                    fall_score *= 0.5
                
                fall_score = min(fall_score, 0.99)
                is_falling = fall_score >= 0.85  # Using threshold 0.85
                
                if is_falling:
                    total_frames_with_fall += 1
                    print(f"\n[FRAME {frame_count}] Person {person_idx + 1}: FALL DETECTED (score={fall_score:.3f})")
                    print(f"  Reasons: {', '.join(reasons)}")
                    print(f"  Features: HWR={HWR:.3f}, Torso={TorsoAngle:.1f}°, H={H:.3f}, FallAngle={FallAngleD:.1f}°")
                
                if frame_count % 50 == 0 and frame_count > 0:
                    print(f"\n[FRAME {frame_count}] Person {person_idx + 1}: stand score={fall_score:.3f}")
                    print(f"  Features: HWR={HWR:.3f}, Torso={TorsoAngle:.1f}°, H={H:.3f}, FallAngle={FallAngleD:.1f}°")
                    print(f"  Reasons: {', '.join(reasons) if reasons else 'none'}")
        
        frame_count += 1
        
        if frame_count % 50 == 0:
            print(f"Processed {frame_count} frames: 1-person={total_frames_with_1_person}, 2-person={total_frames_with_2_people}, fall-frames={total_frames_with_fall}")

except Exception as e:
    print(f"ERROR during processing: {e}")
    import traceback
    traceback.print_exc()

finally:
    cap.release()

print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)
print(f"Total frames processed: {frame_count}")
print(f"Frames with people detected: {len(frames_with_detection)}")
print(f"Frames with 1 person: {total_frames_with_1_person}")
print(f"Frames with 2+ people: {total_frames_with_2_people}")
print(f"Frames with fall detected: {total_frames_with_fall}")
print(f"Fall detection rate: {total_frames_with_fall / max(1, frame_count) * 100:.1f}%")

if frames_with_detection:
    print(f"\nDetection timeline (sample):")
    for frame_num, num_people in frames_with_detection[:10]:
        print(f"  Frame {frame_num}: {num_people} person(s)")
    if len(frames_with_detection) > 10:
        print(f"  ... ({len(frames_with_detection) - 10} more frames with detections)")
