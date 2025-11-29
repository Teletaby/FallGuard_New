#!/usr/bin/env python3
"""
Test the detect_multiple_people function on the uploaded video
"""
import os
import sys
import cv2
import mediapipe as mp
import numpy as np

# Add app to path to import functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from video_utils import detect_multiple_people

def test_detect_multiple_people(video_path):
    """Test the detect_multiple_people function"""
    print(f"\n=== Testing detect_multiple_people ===")
    print(f"Video: {os.path.basename(video_path)}")
    
    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.4
    )
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("ERROR: Cannot open video")
        return
    
    frame_count = 0
    detection_count = 0
    frames_with_people = 0
    total_people_detected = 0
    max_people_in_frame = 0
    
    print("\nProcessing frames...")
    
    while frame_count < 100:  # Test first 100 frames
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Resize to 640x480 as the app does
        frame = cv2.resize(frame, (640, 480))
        
        # Call detect_multiple_people
        try:
            people = detect_multiple_people(frame, pose, use_hog=False)
            
            if people:
                frames_with_people += 1
                num_people = len(people)
                total_people_detected += num_people
                max_people_in_frame = max(max_people_in_frame, num_people)
                detection_count += 1
                
                if frame_count % 30 == 0 or frame_count <= 3:
                    print(f"Frame {frame_count:3d}: DETECTED {num_people} person(s)")
                    for i, p in enumerate(people):
                        print(f"  Person {i+1}: x={p['x']:.0f}, y={p['y']:.0f}, conf={p.get('confidence', 0):.2f}")
            else:
                if frame_count % 30 == 0:
                    print(f"Frame {frame_count:3d}: no detection")
        except Exception as e:
            print(f"Frame {frame_count:3d}: ERROR - {e}")
            import traceback
            traceback.print_exc()
    
    cap.release()
    pose.close()
    
    print(f"\n=== Results ===")
    print(f"Total frames processed: {frame_count}")
    print(f"Frames with detections: {frames_with_people}")
    print(f"Detection rate: {100*frames_with_people/frame_count if frame_count > 0 else 0:.1f}%")
    print(f"Total detections: {detection_count}")
    print(f"Total people detected across all frames: {total_people_detected}")
    print(f"Max people in a single frame: {max_people_in_frame}")

if __name__ == "__main__":
    uploads_path = os.path.join(os.path.dirname(__file__), "app", "uploads")
    if os.path.exists(uploads_path):
        files = [f for f in os.listdir(uploads_path) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
        if files:
            for f in files:
                full_path = os.path.join(uploads_path, f)
                test_detect_multiple_people(full_path)
        else:
            print("No video files found in uploads folder")
    else:
        print(f"Uploads folder not found: {uploads_path}")
