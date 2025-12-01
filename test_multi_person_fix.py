#!/usr/bin/env python3
"""
Test script to verify multi-person detection fixes
"""

import cv2
import time
from app.video_utils import detect_multiple_people

def test_multi_person_detection():
    video_path = "uploads/12.mp4"
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"[TEST] Video: {video_path}")
    print(f"[TEST] Total frames: {total_frames}, FPS: {fps}")
    print(f"[TEST] Sampling every 10 frames for quick test...\n")
    
    frame_count = 0
    people_per_frame = []
    fps_measurements = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Sample every 10 frames for speed
        if frame_count % 10 != 0:
            continue
        
        start_time = time.time()
        people = detect_multiple_people(frame, None, use_hog=False)
        elapsed = time.time() - start_time
        
        measured_fps = 1.0 / elapsed if elapsed > 0 else 0
        fps_measurements.append(measured_fps)
        people_per_frame.append(len(people))
        
        print(f"Frame {frame_count:4d}: {len(people)} people detected, Time: {elapsed*1000:.1f}ms, Est FPS: {measured_fps:.1f}")
    
    cap.release()
    
    # Summary
    print(f"\n[SUMMARY]")
    print(f"Frames analyzed: {frame_count}")
    print(f"Average people per frame: {sum(people_per_frame) / len(people_per_frame) if people_per_frame else 0:.2f}")
    print(f"Max people in single frame: {max(people_per_frame) if people_per_frame else 0}")
    print(f"Average FPS: {sum(fps_measurements) / len(fps_measurements) if fps_measurements else 0:.1f}")
    print(f"Min FPS: {min(fps_measurements) if fps_measurements else 0:.1f}")
    print(f"Max FPS: {max(fps_measurements) if fps_measurements else 0:.1f}")

if __name__ == "__main__":
    test_multi_person_detection()
