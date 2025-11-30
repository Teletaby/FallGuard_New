#!/usr/bin/env python3
"""
Test MediaPipe detection on the uploaded video
"""
import os
import cv2
import mediapipe as mp
import numpy as np

def test_mediapipe_on_video(video_path):
    """Test MediaPipe pose detection on a video"""
    print(f"\n=== Testing MediaPipe on Video ===")
    print(f"Video: {video_path}")
    
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
    
    print("\nProcessing frames...")
    
    while frame_count < 100:  # Test first 100 frames only
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Resize to 640x480 as the app does
        frame = cv2.resize(frame, (640, 480))
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run MediaPipe
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            num_people = 1  # MediaPipe detects one person per pass
            total_people_detected += 1
            frames_with_people += 1
            detection_count += 1
            
            # Print every 30 frames
            if frame_count % 30 == 0:
                print(f"Frame {frame_count:3d}: DETECTED - landmarks={len(results.pose_landmarks.landmark)}")
        else:
            if frame_count % 30 == 0:
                print(f"Frame {frame_count:3d}: no detection")
    
    cap.release()
    pose.close()
    
    print(f"\n=== Results ===")
    print(f"Total frames processed: {frame_count}")
    print(f"Frames with detections: {frames_with_people}")
    print(f"Detection rate: {100*frames_with_people/frame_count if frame_count > 0 else 0:.1f}%")
    print(f"Total detections: {detection_count}")

if __name__ == "__main__":
    uploads_path = os.path.join(os.path.dirname(__file__), "app", "uploads")
    if os.path.exists(uploads_path):
        files = [f for f in os.listdir(uploads_path) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
        if files:
            for f in files:
                full_path = os.path.join(uploads_path, f)
                test_mediapipe_on_video(full_path)
        else:
            print("No video files found in uploads folder")
    else:
        print(f"Uploads folder not found: {uploads_path}")
