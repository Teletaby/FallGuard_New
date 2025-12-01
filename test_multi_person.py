#!/usr/bin/env python3
"""
Test multi-person detection on uploaded video
"""

import cv2
import numpy as np
import os
from app.video_utils import detect_multiple_people

def test_multi_person():
    video_path = "uploads/12.mp4"
    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        return
    
    print(f"\n[TEST] Multi-Person Detection Test")
    print(f"[VIDEO] {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[VIDEO] Total frames: {total_frames}\n")
    
    frame_count = 0
    max_people_frames = {}
    max_people = 0
    
    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (640, 480))
        frame_count += 1
        
        # Detect people
        people = detect_multiple_people(frame, None, use_hog=False)
        
        if len(people) > 0:
            print(f"Frame {frame_count:3d}: Detected {len(people)} person(s)", end="")
            
            # Show details
            for i, person in enumerate(people):
                bbox = person['bbox']
                print(f" | P{i+1}: bbox=({bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f})", end="")
            
            print()
            
            if len(people) > max_people:
                max_people = len(people)
                max_people_frames[len(people)] = frame_count
    
    cap.release()
    
    print(f"\n[SUMMARY]")
    print(f"Frames scanned: {total_frames}")
    print(f"Max people detected simultaneously: {max_people}")
    if max_people > 1:
        print(f"✓ Multi-person detection WORKING")
        for num_people in sorted(max_people_frames.keys(), reverse=True):
            print(f"  - {num_people} people detected at frame {max_people_frames[num_people]}")
    else:
        print(f"✗ Only 1 person detected - check YOLO confidence or room has only 1 person")

if __name__ == "__main__":
    try:
        test_multi_person()
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
