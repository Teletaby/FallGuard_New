#!/usr/bin/env python3
"""Test multi-person skeleton drawing in detail"""

import cv2
import numpy as np
from app.video_utils import detect_multiple_people
import mediapipe as mp

# Load a test frame
test_video = "uploads/1.mp4"
cap = cv2.VideoCapture(test_video)

if not cap.isOpened():
    print(f"Cannot open: {test_video}")
    exit(1)

ret, frame = cap.read()
if not ret:
    print("Cannot read frame")
    exit(1)

frame = cv2.resize(frame, (640, 480))
print(f"Frame shape: {frame.shape}\n")

# Detect people
print("=== DETECTION ===")
people = detect_multiple_people(frame, use_hog=False)
print(f"\nDetected {len(people)} people with adequate keypoints\n")

if len(people) == 0:
    print("No people detected!")
    cap.release()
    exit(1)

# Try to draw all of them
print("=== SKELETON DRAWING ===")
mp_drawing = mp.solutions.drawing_utils
test_frame = frame.copy()
draw_count = 0

colors = [
    (0, 255, 0),      # Green
    (255, 0, 0),      # Blue
    (0, 255, 255),    # Yellow
    (255, 0, 255),    # Magenta
    (255, 255, 0),    # Cyan
]

for idx, person in enumerate(people):
    landmarks = person['landmarks']
    color = colors[idx % len(colors)]
    
    print(f"\nPerson {idx}:")
    print(f"  Landmarks: {type(landmarks)}")
    print(f"  Valid landmarks: {sum(1 for lm in landmarks.landmark if lm.visibility > 0.1)}")
    
    try:
        mp_drawing.draw_landmarks(
            test_frame,
            landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=color, thickness=6, circle_radius=5),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=color, thickness=1)
        )
        draw_count += 1
        print(f"  ✓ Drawn with color {color}")
    except Exception as e:
        print(f"  ✗ Draw failed: {e}")

print(f"\n=== RESULT ===")
print(f"Drew {draw_count} skeletons")
cv2.imwrite("test_multi_skeleton.jpg", test_frame)
print(f"Saved to: test_multi_skeleton.jpg")

cap.release()
