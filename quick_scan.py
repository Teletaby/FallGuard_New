#!/usr/bin/env python3
"""Quick scan for multi-person frames"""

import cv2
from app.video_utils import detect_multiple_people

cap = cv2.VideoCapture("uploads/12.mp4")
frame_count = 0
two_plus_frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (640, 480))
    frame_count += 1
    people = detect_multiple_people(frame, None, use_hog=False)
    
    if len(people) >= 2:
        two_plus_frames.append(frame_count)
        if len(two_plus_frames) <= 5:
            print(f"Frame {frame_count}: {len(people)} people")

cap.release()

print(f"\nTotal frames with 2+ people: {len(two_plus_frames)}")
print(f"Frame numbers: {two_plus_frames}")
