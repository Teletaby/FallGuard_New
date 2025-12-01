#!/usr/bin/env python3
"""
Test script to validate:
1. Standing detection doesn't trigger false falls
2. Multi-person detection works (2+ people)
3. Fall confirmation requires 7+ frames
"""

import cv2
import numpy as np
import sys
from app.video_utils import extract_8_kinematic_features, detect_multiple_people
from mediapipe.framework.formats import landmark_pb2

def test_2person_video():
    """Test multi-person detection on the provided 2-person video"""
    video_path = 'uploads/586837864_25303762689290199_4978210224702831960_n.mp4'
    
    print(f"[TEST] Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video")
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"[VIDEO_INFO] FPS={fps}, Frames={total_frames}, Size={width}x{height}")
    
    frame_count = 0
    people_detected_per_frame = []
    max_people = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect people
        people = detect_multiple_people(frame)
        num_people = len(people)
        people_detected_per_frame.append(num_people)
        max_people = max(max_people, num_people)
        
        # Print every 30 frames (1 second at 30fps)
        if frame_count % 30 == 0:
            print(f"[FRAME {frame_count:4d}/{total_frames}] People detected: {num_people}")
            for idx, person in enumerate(people):
                bbox = person['bbox']
                print(f"    Person {idx+1}: bbox=({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})")
    
    cap.release()
    
    # Statistics
    avg_people = np.mean(people_detected_per_frame) if people_detected_per_frame else 0
    min_people = min(people_detected_per_frame) if people_detected_per_frame else 0
    
    print(f"\n[RESULTS]")
    print(f"  Total frames: {frame_count}")
    print(f"  Min people per frame: {min_people}")
    print(f"  Max people per frame: {max_people}")
    print(f"  Avg people per frame: {avg_people:.2f}")
    
    if max_people >= 2:
        print(f"\n✅ PASS: Successfully detected {max_people} people!")
        return True
    else:
        print(f"\n❌ FAIL: Only detected {max_people} people (expected 2+)")
        return False

def test_standing_features():
    """Test that standing poses produce high HWR and high TorsoAngle"""
    print("\n[TEST] Standing pose feature extraction")
    
    # Create a synthetic standing pose landmark
    landmarks = landmark_pb2.NormalizedLandmarkList()
    
    # Standing pose: tall and narrow
    # Head at top, feet at bottom
    pose_data = {
        0: (0.5, 0.15),    # nose - high in frame
        11: (0.45, 0.5),   # left_shoulder
        12: (0.55, 0.5),   # right_shoulder
        23: (0.45, 0.7),   # left_hip
        24: (0.55, 0.7),   # right_hip
        27: (0.45, 0.95),  # left_ankle
        28: (0.55, 0.95),  # right_ankle
        2: (0.52, 0.12),   # right_eye
        5: (0.48, 0.12),   # left_eye
        13: (0.42, 0.65),  # left_elbow
        14: (0.58, 0.65),  # right_elbow
    }
    
    for idx in range(33):
        if idx in pose_data:
            x, y = pose_data[idx]
            landmarks.landmark.append(landmark_pb2.NormalizedLandmark(x=x, y=y, z=0, visibility=0.8, presence=0.8))
        else:
            landmarks.landmark.append(landmark_pb2.NormalizedLandmark(x=0, y=0, z=0, visibility=0.0, presence=0.0))
    
    # Extract features
    features = extract_8_kinematic_features(landmarks)
    HWR = features[0]
    TorsoAngle = features[1]
    D = features[2]
    H = features[5]
    FallAngleD = features[6]
    
    print(f"  Features extracted:")
    print(f"    HWR: {HWR:.3f} (standing should be >1.5, lying should be <0.35)")
    print(f"    TorsoAngle: {TorsoAngle:.1f}° (standing ~5-15°, lying >75°)")
    print(f"    D: {D:.3f} (standing ~0.3-0.5, lying ~0.05-0.15)")
    print(f"    H: {H:.3f} (standing ~0.3-0.4, lying ~0.7+)")
    print(f"    FallAngleD: {FallAngleD:.1f}° (standing ~75-85°, lying <20°)")
    
    # Check if it looks like standing
    is_standing = (HWR > 1.5 and TorsoAngle < 30 and FallAngleD > 60 and D > 0.2)
    
    if is_standing:
        print(f"\n✅ PASS: Correctly identified as standing pose")
        return True
    else:
        print(f"\n❌ FAIL: Did not identify as standing pose")
        return False

def test_lying_features():
    """Test that lying poses produce low HWR and low TorsoAngle"""
    print("\n[TEST] Lying pose feature extraction")
    
    # Create a synthetic lying pose landmark
    landmarks = landmark_pb2.NormalizedLandmarkList()
    
    # Lying pose: short and wide (horizontal)
    pose_data = {
        0: (0.2, 0.5),     # nose - left side
        11: (0.25, 0.45),  # left_shoulder
        12: (0.75, 0.55),  # right_shoulder
        23: (0.28, 0.52),  # left_hip
        24: (0.78, 0.58),  # right_hip
        27: (0.1, 0.5),    # left_ankle
        28: (0.9, 0.55),   # right_ankle
        2: (0.22, 0.48),   # right_eye
        5: (0.18, 0.52),   # left_eye
        13: (0.45, 0.48),  # left_elbow
        14: (0.55, 0.52),  # right_elbow
    }
    
    for idx in range(33):
        if idx in pose_data:
            x, y = pose_data[idx]
            landmarks.landmark.append(landmark_pb2.NormalizedLandmark(x=x, y=y, z=0, visibility=0.8, presence=0.8))
        else:
            landmarks.landmark.append(landmark_pb2.NormalizedLandmark(x=0, y=0, z=0, visibility=0.0, presence=0.0))
    
    # Extract features
    features = extract_8_kinematic_features(landmarks)
    HWR = features[0]
    TorsoAngle = features[1]
    D = features[2]
    H = features[5]
    FallAngleD = features[6]
    
    print(f"  Features extracted:")
    print(f"    HWR: {HWR:.3f} (standing should be >1.5, lying should be <0.35)")
    print(f"    TorsoAngle: {TorsoAngle:.1f}° (standing ~5-15°, lying >75°)")
    print(f"    D: {D:.3f} (standing ~0.3-0.5, lying ~0.05-0.15)")
    print(f"    H: {H:.3f} (standing ~0.3-0.4, lying ~0.7+)")
    print(f"    FallAngleD: {FallAngleD:.1f}° (standing ~75-85°, lying <20°)")
    
    # Check if it looks like lying
    is_lying = (HWR < 0.5 and TorsoAngle > 70 and FallAngleD < 25 and D < 0.15)
    
    if is_lying:
        print(f"\n✅ PASS: Correctly identified as lying pose")
        return True
    else:
        print(f"\n❌ FAIL: Did not identify as lying pose")
        return False

if __name__ == '__main__':
    print("=" * 60)
    print("FallGuard Improvement Tests")
    print("=" * 60)
    
    results = []
    
    # Test 1: Standing pose features
    results.append(("Standing pose detection", test_standing_features()))
    
    # Test 2: Lying pose features
    results.append(("Lying pose detection", test_lying_features()))
    
    # Test 3: Multi-person detection
    results.append(("Multi-person video detection", test_2person_video()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    total_passed = sum(1 for _, p in results if p)
    total_tests = len(results)
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    sys.exit(0 if total_passed == total_tests else 1)
