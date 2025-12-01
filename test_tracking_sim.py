#!/usr/bin/env python3
"""
Detailed multi-person tracking simulation to debug why only 1 person persists
"""

import cv2
import numpy as np
import os
import time
from app.video_utils import detect_multiple_people

def simulate_tracking():
    video_path = "uploads/12.mp4"
    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        return
    
    print(f"\n[TEST] Multi-Person Tracking Simulation")
    print(f"[VIDEO] {video_path}\n")
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Simulate the tracking state from main.py
    people_trackers = {}
    next_person_id = 1
    person_timeout = 2.0
    
    frame_count = 0
    two_person_frames = []
    
    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (640, 480))
        frame_count += 1
        current_time = time.time()
        
        # Detect people
        people = detect_multiple_people(frame, None, use_hog=False)
        
        if len(people) > 1:
            print(f"\n=== FRAME {frame_count}: YOLO detected {len(people)} people ===")
            
            # Try matching using distance threshold 600px
            for person_idx, person in enumerate(people):
                person_bbox = person['bbox']
                x, y, w, h = person_bbox
                bbox_center = (x + w/2, y + h/2)
                
                # Try to match to existing tracker
                best_match = None
                best_distance = float('inf')
                threshold = 600
                
                for pid, tracker in people_trackers.items():
                    tracker_center = tracker['center']
                    distance = np.sqrt((bbox_center[0] - tracker_center[0])**2 + (bbox_center[1] - tracker_center[1])**2)
                    
                    if distance < threshold and distance < best_distance:
                        best_distance = distance
                        best_match = pid
                
                if best_match is None:
                    # New person
                    person_id = next_person_id
                    next_person_id += 1
                    print(f"  Person {person_idx}: NEW ID #{person_id}, bbox={person_bbox}, center={bbox_center}")
                else:
                    person_id = best_match
                    print(f"  Person {person_idx}: Matched to ID #{person_id}, distance={best_distance:.1f}px, bbox={person_bbox}, center={bbox_center}")
                
                # Update tracker
                people_trackers[person_id] = {
                    'center': bbox_center,
                    'bbox': person_bbox,
                    'last_seen': current_time
                }
            
            # Show all tracked people
            print(f"  Active trackers: {list(people_trackers.keys())}")
            two_person_frames.append(frame_count)
        
        # Clean up stale trackers
        stale = []
        for pid, tracker in people_trackers.items():
            if current_time - tracker['last_seen'] > person_timeout:
                stale.append(pid)
        
        for pid in stale:
            del people_trackers[pid]
        
        if frame_count > total_frames:
            break
    
    cap.release()
    
    print(f"\n[SUMMARY]")
    print(f"Frames with 2+ people detected by YOLO: {len(two_person_frames)}")
    if two_person_frames:
        print(f"Frame numbers: {two_person_frames}")
    else:
        print("No frames with 2+ people")

if __name__ == "__main__":
    try:
        simulate_tracking()
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
