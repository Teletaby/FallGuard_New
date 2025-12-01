#!/usr/bin/env python3
"""Test skeleton drawing for the two-person video"""
import cv2
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.video_utils import detect_multiple_people

# Video path
video_path = r"uploads/586837864_25303762689290199_4978210224702831960_n.mp4"

if not os.path.exists(video_path):
    print(f"ERROR: Video not found at {video_path}")
    print(f"Current dir: {os.getcwd()}")
    print(f"Files in uploads/: {os.listdir('uploads') if os.path.exists('uploads') else 'uploads dir not found'}")
    sys.exit(1)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"ERROR: Cannot open video {video_path}")
    sys.exit(1)

print(f"Video opened: {video_path}")
print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
print(f"Total frames: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")

frame_count = 0
for _ in range(100):  # Check first 100 frames
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    frame = cv2.resize(frame, (640, 480))
    
    print(f"\n[FRAME {frame_count}]")
    people = detect_multiple_people(frame, None, use_hog=False)
    print(f"  Detected {len(people)} people")
    
    for person_idx, person in enumerate(people):
        bbox = person['bbox']
        landmarks = person['landmarks']
        x, y = person['x'], person['y']
        
        print(f"  Person {person_idx}: center=({x:.0f},{y:.0f}), bbox={bbox}")
        
        if landmarks and hasattr(landmarks, 'landmark'):
            valid_keypoints = sum(1 for lm in landmarks.landmark if lm.visibility > 0.2)
            print(f"    Landmarks: {len(landmarks.landmark)} total, {valid_keypoints} valid (visibility>0.2)")
            
            # Show first few landmarks
            for i in range(min(5, len(landmarks.landmark))):
                lm = landmarks.landmark[i]
                print(f"      [{i}] vis={lm.visibility:.2f} x={lm.x:.2f} y={lm.y:.2f} z={lm.z:.2f}")

cap.release()
print(f"\nProcessed {frame_count} frames")
