#!/usr/bin/env python3
"""
Test real video with the updated fall detection logic
"""
import cv2
import numpy as np
from app.video_utils import detect_multiple_people

def test_real_video():
    """Test 2-person video with actual detection"""
    video_path = 'uploads/586837864_25303762689290199_4978210224702831960_n.mp4'
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Testing: {video_path}")
    print(f"FPS: {fps}, Total frames: {total_frames}")
    print()
    
    frame_count = 0
    people_counts = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect people
        people = detect_multiple_people(frame)
        num_people = len(people)
        people_counts.append(num_people)
        
        # Print summary every second (30 frames)
        if frame_count % 30 == 0 or frame_count == total_frames:
            print(f"Frame {frame_count:3d}/{total_frames}: {num_people} people detected")
    
    cap.release()
    
    # Calculate statistics
    max_people = max(people_counts) if people_counts else 0
    avg_people = np.mean(people_counts) if people_counts else 0
    
    print()
    print(f"Results:")
    print(f"  Total frames: {frame_count}")
    print(f"  Max people detected: {max_people}")
    print(f"  Average people per frame: {avg_people:.2f}")
    print()
    
    if max_people >= 2:
        print("✅ SUCCESS: Multi-person detection working!")
        print("   The system can now detect 2+ people in the same frame.")
        return True
    else:
        print("❌ ISSUE: Only detected 1 person at a time")
        return False

if __name__ == '__main__':
    success = test_real_video()
    exit(0 if success else 1)
