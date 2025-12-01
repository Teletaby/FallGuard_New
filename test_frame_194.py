#!/usr/bin/env python3
"""
Test what happens at frame 194 where 2 people are detected
"""

import cv2
import numpy as np
import os
from app.video_utils import detect_multiple_people

def test_frame_194():
    video_path = "uploads/12.mp4"
    cap = cv2.VideoCapture(video_path)
    
    # Jump to frame 194
    cap.set(cv2.CAP_PROP_POS_FRAMES, 193)
    
    print("[TEST] Analyzing frames 194-200 where 2 people appear\n")
    
    for i in range(8):
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (640, 480))
        frame_num = 194 + i
        
        # Detect people
        people = detect_multiple_people(frame, None, use_hog=False)
        
        print(f"Frame {frame_num}: {len(people)} people detected")
        for p_idx, person in enumerate(people):
            bbox = person['bbox']
            x, y, w, h = bbox
            center = (x + w/2, y + h/2)
            print(f"  Person {p_idx}: bbox=({x:.0f},{y:.0f},{w:.0f},{h:.0f}), center={center}")
    
    cap.release()

if __name__ == "__main__":
    try:
        test_frame_194()
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
