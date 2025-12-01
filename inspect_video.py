#!/usr/bin/env python3
"""
Detailed frame inspection to see what's in the video
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os

def inspect_video_frames():
    """Inspect multiple frames from the video"""
    video_path = "uploads/12.mp4"
    
    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        return
    
    print(f"[INFO] Inspecting: {video_path}\n")
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"[INFO] Total frames: {total_frames}")
    print(f"[INFO] FPS: {fps}")
    print(f"[INFO] Duration: {total_frames/fps:.1f}s\n")
    
    # Load YOLOv11
    model = YOLO('yolo11n-pose.pt')
    
    # Check every 10th frame or every 30 frames max
    frames_to_check = min(5, total_frames // 30)
    frame_indices = [int(i * total_frames / frames_to_check) for i in range(frames_to_check)]
    
    print(f"[INFO] Checking {len(frame_indices)} frames: {frame_indices}\n")
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        frame = cv2.resize(frame, (640, 480))
        
        print(f"Frame {frame_idx}:")
        
        # Detect with YOLOv11
        results = model(frame, conf=0.25, iou=0.50, verbose=False)
        
        if results:
            result = results[0]
            num_people = len(result.boxes) if result.boxes else 0
            print(f"  YOLOv11 detected: {num_people} person(s)")
            
            if result.boxes:
                for i, box in enumerate(result.boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf)
                    print(f"    Person {i+1}: bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}), conf={conf:.3f}")
        else:
            print("  No detections")
        
        # Also try lower confidence to see if there are people being filtered out
        results_low = model(frame, conf=0.10, iou=0.50, verbose=False)
        if results_low:
            result_low = results_low[0]
            num_people_low = len(result_low.boxes) if result_low.boxes else 0
            if num_people_low > num_people if result_low.boxes else 0:
                print(f"    [NOTE] With conf=0.10: {num_people_low} detected")
        
        print()
    
    cap.release()

if __name__ == "__main__":
    try:
        inspect_video_frames()
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
