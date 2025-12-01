#!/usr/bin/env python3
"""
Debug script to test multi-person detection on actual video files
"""

import cv2
import numpy as np
from app.video_utils import detect_multiple_people

def test_multi_person_on_video(video_path):
    """Test multi-person detection on a video file"""
    print(f"\n{'='*70}")
    print(f"Testing multi-person detection on: {video_path}")
    print(f"{'='*70}\n")
    
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"✗ Failed to open video: {video_path}")
            return False
        
        frame_count = 0
        detection_results = []
        
        while frame_count < 300:  # Test first 300 frames (~10 seconds at 30fps)
            ret, frame = cap.read()
            
            if not ret:
                print(f"✓ Reached end of video at frame {frame_count}")
                break
            
            frame = cv2.resize(frame, (640, 480))
            
            # Detect people
            people = detect_multiple_people(frame, None, use_hog=False)
            
            detection_results.append({
                'frame': frame_count,
                'num_people': len(people),
                'people': people
            })
            
            if frame_count % 30 == 0:
                print(f"[Frame {frame_count:3d}] Detected {len(people)} person(s)")
                for idx, person in enumerate(people):
                    conf = person.get('confidence', 0.0)
                    bbox = person.get('bbox', (0,0,0,0))
                    print(f"           Person {idx+1}: confidence={conf:.3f}, bbox={bbox}")
            
            frame_count += 1
        
        cap.release()
        
        # Analyze results
        total_detections = sum(1 for r in detection_results)
        avg_people = np.mean([r['num_people'] for r in detection_results]) if detection_results else 0
        max_people = max([r['num_people'] for r in detection_results]) if detection_results else 0
        
        print(f"\n{'='*70}")
        print(f"[SUMMARY]")
        print(f"  Frames analyzed: {len(detection_results)}")
        print(f"  Average people per frame: {avg_people:.2f}")
        print(f"  Maximum people in single frame: {max_people}")
        print(f"  Frames with 0 people: {sum(1 for r in detection_results if r['num_people'] == 0)}")
        print(f"  Frames with 1 person: {sum(1 for r in detection_results if r['num_people'] == 1)}")
        print(f"  Frames with 2+ people: {sum(1 for r in detection_results if r['num_people'] >= 2)}")
        print(f"{'='*70}")
        
        if max_people >= 2:
            print(f"✓ Multi-person detection working (max {max_people} people)")
            return True
        else:
            print(f"✗ Multi-person detection not working (max {max_people} people)")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import os
    import sys
    
    # Look for video files in uploads folder
    upload_dir = 'uploads'
    
    if len(sys.argv) > 1:
        # Test specific video if provided
        video_path = sys.argv[1]
        test_multi_person_on_video(video_path)
    elif os.path.exists(upload_dir):
        # Test all videos in uploads folder
        videos = [f for f in os.listdir(upload_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        if videos:
            print(f"\nFound {len(videos)} video(s) in {upload_dir}/")
            for video in videos[:3]:  # Test first 3 videos
                video_path = os.path.join(upload_dir, video)
                test_multi_person_on_video(video_path)
        else:
            print(f"\n✗ No videos found in {upload_dir}/")
            print("\nUsage: python debug_multi_person.py [video_path]")
    else:
        print(f"\n✗ Directory '{upload_dir}' not found")
        print("\nUsage: python debug_multi_person.py [video_path]")
