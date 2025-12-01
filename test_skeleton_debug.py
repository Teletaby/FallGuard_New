#!/usr/bin/env python3
"""Quick test to verify skeleton drawing for multiple people"""

import cv2
from app.video_utils import detect_multiple_people

def test_skeleton():
    cap = cv2.VideoCapture('uploads/12.mp4')
    if not cap.isOpened():
        print("[ERROR] Cannot open video")
        return
    
    # Read first frame with people detected
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        people = detect_multiple_people(frame)
        
        if len(people) >= 2:
            print(f"\n[FOUND] Frame {frame_num}: {len(people)} people detected")
            for i, person in enumerate(people):
                landmarks = person['landmarks']
                print(f"  Person {i}:")
                print(f"    - landmarks type: {type(landmarks)}")
                print(f"    - has 'landmark' attr: {hasattr(landmarks, 'landmark')}")
                if hasattr(landmarks, 'landmark'):
                    print(f"    - num landmarks: {len(landmarks.landmark)}")
                    valid_lms = sum(1 for lm in landmarks.landmark if lm.visibility > 0.3)
                    print(f"    - valid landmarks (vis>0.3): {valid_lms}")
            break
        
        if frame_num >= 100:
            print("[INFO] Checked 100 frames, no 2-person detection found")
            break
    
    cap.release()

if __name__ == "__main__":
    test_skeleton()
