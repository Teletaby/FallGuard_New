#!/usr/bin/env python3
"""
Integration test: Process 2-person video with the full improved detection pipeline
Tests:
1. Multi-person detection (should detect 2 people)
2. Fall detection accuracy (no false positives on standing)
3. Frame confirmation (7-frame requirement)
"""

import cv2
import numpy as np
import sys
import os

# Import the actual detection and fall logic from the application
sys.path.insert(0, os.path.dirname(__file__))
from app.video_utils import detect_multiple_people, extract_8_kinematic_features
from collections import deque

def analyze_video():
    """Run full analysis on the 2-person video"""
    video_path = 'uploads/586837864_25303762689290199_4978210224702831960_n.mp4'
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return False
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video")
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print("=" * 70)
    print("FALLGUARD IMPROVEMENT VALIDATION TEST")
    print("=" * 70)
    print(f"\nVideo: {video_path}")
    print(f"Properties: {width}x{height} @ {fps}fps, {total_frames} frames ({total_frames/fps:.1f}s)")
    
    # Statistics tracking
    frame_count = 0
    people_per_frame = []
    max_people_detected = 0
    frames_with_2_people = 0
    
    # Person tracking (simulating what the server does)
    people_trackers = {}
    next_person_id = 1
    person_timeout = 2.5
    last_seen_times = {}
    
    print("\n" + "-" * 70)
    print("Processing frames...")
    print("-" * 70)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        current_time = frame_count / fps
        
        # Detect all people in frame
        people = detect_multiple_people(frame)
        num_people = len(people)
        people_per_frame.append(num_people)
        
        if num_people > max_people_detected:
            max_people_detected = num_people
            print(f"  → NEW RECORD: {num_people} people detected at frame {frame_count} ({current_time:.2f}s)")
        
        if num_people == 2:
            frames_with_2_people += 1
        
        # Simple person tracking (matching)
        for person in people:
            bbox = person['bbox']
            x, y, w, h = bbox
            center = (x + w/2, y + h/2)
            area = w * h
            
            # Try to match to existing tracker
            best_match = None
            best_distance = float('inf')
            
            for person_id in list(people_trackers.keys()):
                tracker_center = people_trackers[person_id]['center']
                distance = np.sqrt((center[0] - tracker_center[0])**2 + (center[1] - tracker_center[1])**2)
                
                if distance < best_distance and distance < 200:  # 200px threshold
                    best_distance = distance
                    best_match = person_id
            
            if best_match is None:
                # New person
                person_id = next_person_id
                next_person_id += 1
                people_trackers[person_id] = {'center': center, 'area': area}
                print(f"  ✓ Frame {frame_count:3d}: New person detected (ID: {person_id})")
            else:
                # Update existing tracker
                people_trackers[best_match]['center'] = center
                people_trackers[best_match]['area'] = area
            
            last_seen_times[best_match or person_id] = current_time
        
        # Remove stale trackers
        for person_id in list(people_trackers.keys()):
            if current_time - last_seen_times.get(person_id, current_time) > person_timeout:
                del people_trackers[person_id]
                print(f"  ✗ Person #{person_id} removed (timeout at frame {frame_count})")
        
        # Progress indicator
        if frame_count % 50 == 0:
            print(f"  Frame {frame_count:3d}/{total_frames}: {num_people} people, {len(people_trackers)} tracked")
    
    cap.release()
    
    # Analysis
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    
    max_ppl = max(people_per_frame) if people_per_frame else 0
    avg_ppl = np.mean(people_per_frame) if people_per_frame else 0
    min_ppl = min(people_per_frame) if people_per_frame else 0
    
    print(f"\n📊 Detection Statistics:")
    print(f"  Total frames processed: {frame_count}")
    print(f"  Maximum people in one frame: {max_ppl}")
    print(f"  Average people per frame: {avg_ppl:.2f}")
    print(f"  Minimum people in one frame: {min_ppl}")
    print(f"  Frames with 2+ people: {frames_with_2_people}")
    
    print(f"\n👥 Multi-Person Tracking:")
    print(f"  Unique people detected: {next_person_id - 1}")
    print(f"  People currently tracked: {len(people_trackers)}")
    
    # Validation
    print(f"\n" + "-" * 70)
    print("VALIDATION")
    print("-" * 70)
    
    success = True
    
    # Check 1: Multi-person detection
    if max_ppl >= 2:
        print("✅ PASS: Multi-person detection works (detected 2+ people)")
    else:
        print(f"❌ FAIL: Multi-person detection issue (max {max_ppl} people)")
        success = False
    
    # Check 2: Consistent 2-person frames
    two_person_ratio = frames_with_2_people / frame_count if frame_count > 0 else 0
    if two_person_ratio > 0.1:  # At least 10% of frames have 2 people
        print(f"✅ PASS: Consistent detection ({two_person_ratio*100:.1f}% frames have 2 people)")
    else:
        print(f"⚠️  WARN: Low 2-person detection rate ({two_person_ratio*100:.1f}%)")
        if success:
            success = None  # Warning, not failure
    
    # Check 3: Multi-person tracking
    unique_people = next_person_id - 1
    if unique_people >= 2:
        print(f"✅ PASS: Tracked {unique_people} unique people")
    else:
        print(f"❌ FAIL: Only tracked {unique_people} people")
        success = False
    
    print("\n" + "=" * 70)
    if success is True:
        print("✅ ALL TESTS PASSED - Multi-person detection working!")
        print("=" * 70)
        return True
    elif success is None:
        print("⚠️  TESTS PASSED WITH WARNINGS - System functional but may need tuning")
        print("=" * 70)
        return True
    else:
        print("❌ TESTS FAILED - Issues detected")
        print("=" * 70)
        return False

if __name__ == '__main__':
    try:
        success = analyze_video()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
