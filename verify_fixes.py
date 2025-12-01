#!/usr/bin/env python3
"""
Comprehensive test of all fixes
"""

import sys
import os

print("\n" + "="*70)
print("FALLGUARD FIX VERIFICATION TEST")
print("="*70 + "\n")

# Test 1: Check bounding box removal
print("[TEST 1] Checking bounding box code removal...")
try:
    with open('main.py', 'r') as f:
        content = f.read()
        
    # Check if old bounding box drawing code is removed
    if 'Draw bounding boxes for all tracked people' in content:
        print("✗ FAIL: Bounding box drawing code still exists")
    else:
        print("✓ PASS: Bounding box drawing code removed")
        
    # Check if skeleton drawing is present
    if 'Draw skeleton for all tracked people' in content:
        print("✓ PASS: Skeleton drawing code present")
    else:
        print("✗ FAIL: Skeleton drawing code not found")
except Exception as e:
    print(f"✗ ERROR: {e}")

# Test 2: Check detection interval
print("\n[TEST 2] Checking detection interval...")
try:
    with open('main.py', 'r') as f:
        content = f.read()
    
    # Check if pose_process_interval is 1
    if '"pose_process_interval": 1' in content or "'pose_process_interval': 1" in content:
        print("✓ PASS: Detection interval set to 1 (every frame)")
    else:
        print("✗ FAIL: Detection interval not set to 1")
        
    if 'Process pose every frame (required for reliable multi-person)' in content:
        print("✓ PASS: Multi-person detection requirement documented")
    else:
        print("⚠ WARNING: Multi-person comment not found")
except Exception as e:
    print(f"✗ ERROR: {e}")

# Test 3: Check fall detection thresholds
print("\n[TEST 3] Checking fall detection thresholds...")
try:
    from main import CameraProcessor
    
    # Create a dummy processor to check the thresholds
    proc = CameraProcessor("test_cam", 0, "test", device=None)
    
    # Create test feature vector (standing)
    import numpy as np
    standing_features = np.array([1.2, 5, -0.35, 0, 0, 0.55, 85, 0], dtype=np.float32)
    
    is_fall, prob = proc._predict_fall_for_person(1, standing_features)
    
    if is_fall:
        print(f"✗ FAIL: Standing detected as fall (prob={prob:.3f})")
    else:
        print(f"✓ PASS: Standing correctly identified as normal (prob={prob:.3f})")
        
    # Test lying down
    lying_features = np.array([0.35, 88, 0.15, 0, 0, 0.85, 8, 0], dtype=np.float32)
    is_fall, prob = proc._predict_fall_for_person(2, lying_features)
    
    if is_fall:
        print(f"✓ PASS: Lying down detected as fall (prob={prob:.3f})")
    else:
        print(f"✗ FAIL: Lying down not detected (prob={prob:.3f})")
        
except Exception as e:
    print(f"✗ ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Check confidence thresholds
print("\n[TEST 4] Checking YOLOv11 confidence settings...")
try:
    with open('app/video_utils.py', 'r') as f:
        content = f.read()
    
    if 'conf=0.25' in content:
        print("✓ PASS: YOLOv11 confidence set to 0.25")
    else:
        print("⚠ WARNING: YOLOv11 confidence not found or different value")
        
    if 'iou=0.50' in content:
        print("✓ PASS: YOLOv11 IoU set to 0.50")
    else:
        print("⚠ WARNING: YOLOv11 IoU not found or different value")
        
    if 'conf_kpt > 0.5' in content:
        print("✓ PASS: Keypoint confidence threshold set to 0.5")
    else:
        print("⚠ WARNING: Keypoint confidence threshold different")
        
except Exception as e:
    print(f"✗ ERROR: {e}")

# Test 5: Check confirmation frames
print("\n[TEST 5] Checking fall confirmation frames...")
try:
    with open('main.py', 'r') as f:
        content = f.read()
    
    if '>= 7' in content and 'fall_states[person_id' in content:
        print("✓ PASS: Fall confirmation set to 7 frames")
    else:
        print("⚠ WARNING: Fall confirmation frames may not be 7")
        
except Exception as e:
    print(f"✗ ERROR: {e}")

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70 + "\n")
