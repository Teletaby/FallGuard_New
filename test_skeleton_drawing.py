#!/usr/bin/env python3
"""Quick test to verify skeleton drawing functionality"""

import cv2
import numpy as np
from app.video_utils import detect_multiple_people, create_mediapipe_landmarks_from_yolov11
import mediapipe as mp

# Load a test frame
test_video = "uploads/1.mp4"  # Use actual test video
cap = cv2.VideoCapture(test_video)

if not cap.isOpened():
    print(f"Cannot open video: {test_video}")
    exit(1)

ret, frame = cap.read()
if not ret:
    print("Cannot read frame")
    exit(1)

frame = cv2.resize(frame, (640, 480))
print(f"Frame shape: {frame.shape}")

# Detect people
print("\n=== DETECTION PHASE ===")
people = detect_multiple_people(frame, use_hog=False)
print(f"Detected {len(people)} people\n")

if len(people) == 0:
    print("No people detected!")
    cap.release()
    exit(1)

# Test skeleton drawing
print("=== SKELETON DRAWING TEST ===")
mp_drawing = mp.solutions.drawing_utils

for person_idx, person in enumerate(people):
    landmarks = person['landmarks']
    print(f"\nPerson {person_idx}:")
    print(f"  Landmarks type: {type(landmarks)}")
    print(f"  Has 'landmark' attr: {hasattr(landmarks, 'landmark')}")
    
    if hasattr(landmarks, 'landmark'):
        print(f"  Number of landmarks: {len(landmarks.landmark)}")
        valid_count = sum(1 for lm in landmarks.landmark if lm.visibility > 0.1)
        print(f"  Valid landmarks (visibility > 0.1): {valid_count}")
        
        # Try to draw
        try:
            test_frame = frame.copy()
            mp_drawing.draw_landmarks(
                test_frame,
                landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=6, circle_radius=5),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
            )
            print(f"  ✓ Draw successful!")
            
            # Save test frame
            output_path = f"test_skeleton_{person_idx}.jpg"
            cv2.imwrite(output_path, test_frame)
            print(f"  Saved to: {output_path}")
            
        except Exception as e:
            print(f"  ✗ Draw failed: {type(e).__name__}: {e}")

cap.release()
print("\n=== TEST COMPLETE ===")
