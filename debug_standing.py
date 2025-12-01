#!/usr/bin/env python3
"""
Debug script to test standing detection feature extraction
"""

import numpy as np
from app.video_utils import extract_8_kinematic_features
from mediapipe.framework.formats import landmark_pb2

# Create a mock standing pose
def create_standing_landmarks():
    """Create realistic landmarks for standing pose"""
    landmarks = landmark_pb2.NormalizedLandmarkList()
    landmark_list = [landmark_pb2.NormalizedLandmark(x=0, y=0, z=0, visibility=0, presence=0) for _ in range(33)]
    
    # Standing pose - body is vertical
    # Head/Nose
    landmark_list[0] = landmark_pb2.NormalizedLandmark(x=0.5, y=0.2, z=0, visibility=0.9, presence=0.9)  # nose
    
    # Eyes
    landmark_list[2] = landmark_pb2.NormalizedLandmark(x=0.48, y=0.18, z=0, visibility=0.8, presence=0.8)  # left_eye
    landmark_list[5] = landmark_pb2.NormalizedLandmark(x=0.52, y=0.18, z=0, visibility=0.8, presence=0.8)  # right_eye
    
    # Shoulders (horizontal, high up on frame)
    landmark_list[11] = landmark_pb2.NormalizedLandmark(x=0.4, y=0.3, z=0, visibility=0.95, presence=0.95)  # left_shoulder
    landmark_list[12] = landmark_pb2.NormalizedLandmark(x=0.6, y=0.3, z=0, visibility=0.95, presence=0.95)  # right_shoulder
    
    # Hips (below shoulders, still high)
    landmark_list[23] = landmark_pb2.NormalizedLandmark(x=0.42, y=0.55, z=0, visibility=0.9, presence=0.9)  # left_hip
    landmark_list[24] = landmark_pb2.NormalizedLandmark(x=0.58, y=0.55, z=0, visibility=0.9, presence=0.9)  # right_hip
    
    # Knees
    landmark_list[25] = landmark_pb2.NormalizedLandmark(x=0.4, y=0.75, z=0, visibility=0.85, presence=0.85)  # left_knee
    landmark_list[26] = landmark_pb2.NormalizedLandmark(x=0.6, y=0.75, z=0, visibility=0.85, presence=0.85)  # right_knee
    
    # Ankles (low - standing on ground)
    landmark_list[27] = landmark_pb2.NormalizedLandmark(x=0.4, y=0.95, z=0, visibility=0.8, presence=0.8)  # left_ankle
    landmark_list[28] = landmark_pb2.NormalizedLandmark(x=0.6, y=0.95, z=0, visibility=0.8, presence=0.8)  # right_ankle
    
    # Elbows and wrists
    landmark_list[13] = landmark_pb2.NormalizedLandmark(x=0.35, y=0.5, z=0, visibility=0.8, presence=0.8)   # left_elbow
    landmark_list[14] = landmark_pb2.NormalizedLandmark(x=0.65, y=0.5, z=0, visibility=0.8, presence=0.8)   # right_elbow
    landmark_list[15] = landmark_pb2.NormalizedLandmark(x=0.3, y=0.65, z=0, visibility=0.75, presence=0.75) # left_wrist
    landmark_list[16] = landmark_pb2.NormalizedLandmark(x=0.7, y=0.65, z=0, visibility=0.75, presence=0.75) # right_wrist
    
    for lm in landmark_list:
        landmarks.landmark.append(lm)
    
    return landmarks

def test_standing():
    print("="*70)
    print("STANDING POSE TEST")
    print("="*70)
    
    standing_landmarks = create_standing_landmarks()
    features = extract_8_kinematic_features(standing_landmarks)
    
    print("\n[FEATURES] Standing pose:")
    print(f"  HWR (Height-Width Ratio): {features[0]:.3f}")
    print(f"  TorsoAngle: {features[1]:.1f}°")
    print(f"  D (Head-Hip vertical): {features[2]:.3f}")
    print(f"  P40: {features[3]:.3f}")
    print(f"  HipVx: {features[4]:.3f}")
    print(f"  H (Hip height): {features[5]:.3f}")
    print(f"  FallAngleD: {features[6]:.1f}°")
    print(f"  HipVy: {features[7]:.3f}")
    
    # Calculate fall score using new thresholds
    HWR = features[0]
    TorsoAngle = features[1]
    H = features[5]
    FallAngleD = features[6]
    
    fall_score = 0.0
    
    print("\n[SCORING] Fall detection calculation:")
    
    # HWR threshold
    if 0.0 < HWR < 0.45:
        fall_score += 0.35
        print(f"  HWR {HWR:.3f} < 0.45: +0.35 (score={fall_score:.2f})")
        if HWR < 0.30:
            fall_score += 0.30
            print(f"    HWR {HWR:.3f} < 0.30: +0.30 (score={fall_score:.2f})")
    else:
        print(f"  HWR {HWR:.3f} >= 0.45: no score (normal)")
    
    # TorsoAngle threshold
    if TorsoAngle > 65:
        fall_score += 0.30
        print(f"  TorsoAngle {TorsoAngle:.1f}° > 65°: +0.30 (score={fall_score:.2f})")
        if TorsoAngle > 78:
            fall_score += 0.15
            print(f"    TorsoAngle {TorsoAngle:.1f}° > 78°: +0.15 (score={fall_score:.2f})")
    else:
        print(f"  TorsoAngle {TorsoAngle:.1f}° <= 65°: no score (normal)")
    
    # H threshold
    if H > 0.72:
        fall_score += 0.08
        print(f"  H {H:.3f} > 0.72: +0.08 (score={fall_score:.2f})")
        if H > 0.82:
            fall_score += 0.12
            print(f"    H {H:.3f} > 0.82: +0.12 (score={fall_score:.2f})")
    else:
        print(f"  H {H:.3f} <= 0.72: no score (normal)")
    
    # FallAngleD threshold
    if FallAngleD < 20:
        fall_score += 0.35
        print(f"  FallAngleD {FallAngleD:.1f}° < 20°: +0.35 (score={fall_score:.2f})")
        if FallAngleD < 10:
            fall_score += 0.18
            print(f"    FallAngleD {FallAngleD:.1f}° < 10°: +0.18 (score={fall_score:.2f})")
    else:
        print(f"  FallAngleD {FallAngleD:.1f}° >= 20°: no score (normal)")
    
    print(f"\n[RESULT] Total fall score: {fall_score:.2f} (threshold: 0.75)")
    
    if fall_score >= 0.75:
        print("  ❌ DETECTED AS FALL (INCORRECT!)")
    else:
        print("  ✓ Correctly identified as NORMAL")
    
    return fall_score < 0.75

if __name__ == "__main__":
    success = test_standing()
    print("\n" + "="*70)
    if success:
        print("✓ Standing detection works correctly!")
    else:
        print("✗ Standing incorrectly detected as fall")
    print("="*70)
