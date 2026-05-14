#!/usr/bin/env python3
"""
Find frames with falls in the video - IMPROVED MULTI-PERSON TRACKING
"""

import cv2
import numpy as np
import os
from app.video_utils import detect_multiple_people, extract_8_kinematic_features

def find_falls():
    video_path = "uploads/12.mp4"
    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"[VIDEO] Total frames: {total_frames}")
    print(f"[VIDEO] FPS: {fps}, Resolution: {frame_width}x{frame_height}")
    print(f"[VIDEO] Scanning entire video for fall events\n")
    
    frame_count = 0
    person_fall_states = {}
    person_tracker = {}  # Track person ID -> last known position
    next_person_id = 1
    GLOBAL_SETTINGS = {"fall_threshold": 0.90}  # Higher threshold for fewer false positives
    
    fall_confirmed_frames = []
    PERSON_TIMEOUT = 30  # Frames before considering person as "gone"
    POSITION_THRESHOLD = 100  # pixels - max distance to match same person
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect people (no need to resize - video_utils handles it)
        people = detect_multiple_people(frame, None, use_hog=False)
        
        # Timeout detection: mark people as gone if not seen
        for person_id in list(person_tracker.keys()):
            if frame_count - person_tracker[person_id]['last_seen'] > PERSON_TIMEOUT:
                print(f"[TRACKING] Person {person_id} timeout (not seen for {PERSON_TIMEOUT} frames)")
                if person_id in person_fall_states:
                    del person_fall_states[person_id]
                del person_tracker[person_id]
        
        if not people:
            print(f"[DEBUG] Frame {frame_count}: No people detected")
            continue
        
        print(f"[DEBUG] Frame {frame_count}: Detected {len(people)} people(s)")
        
        # Match detected people to existing tracked people
        matched_tracked = set()
        person_id_map = {}  # detected_idx -> person_id
        
        for det_idx, detected_person in enumerate(people):
            det_x = detected_person['x']
            det_y = detected_person['y']
            
            # Find closest tracked person
            best_match_id = None
            best_distance = float('inf')
            
            for tracked_id, tracked_info in person_tracker.items():
                if tracked_id in matched_tracked:
                    continue  # Already matched in this frame
                
                track_x = tracked_info['last_x']
                track_y = tracked_info['last_y']
                
                # Euclidean distance between detected and tracked position
                distance = np.sqrt((det_x - track_x)**2 + (det_y - track_y)**2)
                
                if distance < POSITION_THRESHOLD and distance < best_distance:
                    best_distance = distance
                    best_match_id = tracked_id
            
            if best_match_id is not None:
                # Match to existing person
                person_id_map[det_idx] = best_match_id
                matched_tracked.add(best_match_id)
                person_tracker[best_match_id]['last_x'] = det_x
                person_tracker[best_match_id]['last_y'] = det_y
                person_tracker[best_match_id]['last_seen'] = frame_count
                print(f"[TRACKING] Matched detected person {det_idx} to tracked person {best_match_id} (dist={best_distance:.1f}px)")
            else:
                # New person
                person_id_map[det_idx] = next_person_id
                person_tracker[next_person_id] = {
                    'last_x': det_x,
                    'last_y': det_y,
                    'last_seen': frame_count
                }
                print(f"[TRACKING] NEW PERSON {next_person_id} detected at ({det_x:.0f}, {det_y:.0f})")
                next_person_id += 1
        
        # Process each detected person
        for person_idx, person in enumerate(people):
            person_id = person_id_map[person_idx]
            
            if person_id not in person_fall_states:
                person_fall_states[person_id] = {'frames': 0}
            
            landmarks = person['landmarks']
            try:
                features_8 = extract_8_kinematic_features(landmarks)
            except Exception as e:
                print(f"[ERROR] Person {person_id}: Feature extraction failed: {e}")
                continue
            
            if np.any(features_8 != 0):
                HWR = features_8[0]
                TorsoAngle = features_8[1]
                H = features_8[5]
                FallAngleD = features_8[6]
                
                # Calculate fall score with more conservative thresholds
                fall_score = 0.0
                fall_indicators = 0
                
                # ULTRA-CONSERVATIVE THRESHOLDS - Only detect clear falls, not standing/sitting
                # Standing: HWR ~2.5-4.0, Sitting: HWR ~1.0-2.0, Lying: HWR ~0.2-0.5
                if 0.0 < HWR < 0.30:  # Only VERY flat poses (lying down)
                    fall_score += 0.25
                    fall_indicators += 1
                    if HWR < 0.20:    # Extremely flat (definitely lying)
                        fall_score += 0.30
                        fall_indicators += 1
                
                # TorsoAngle: standing ~5-15°, sitting ~30-50°, lying >75°
                if TorsoAngle > 80:   # Only VERY tilted poses (>80°)
                    fall_score += 0.25
                    fall_indicators += 1
                    if TorsoAngle > 88: # Severely tilted (>88°)
                        fall_score += 0.25
                        fall_indicators += 1
                
                # H (hip height): standing ~0.3-0.4, sitting ~0.5-0.6, lying ~0.8+
                if H > 0.85:          # Only VERY low positions (near bottom of frame)
                    fall_score += 0.20
                    fall_indicators += 1
                    if H > 0.90:      # Extremely low
                        fall_score += 0.15
                        fall_indicators += 1
                
                # FallAngleD: standing ~75-85°, lying <15°
                if FallAngleD < 15:   # Only near-horizontal (lying flat)
                    fall_score += 0.25
                    fall_indicators += 1
                    if FallAngleD < 8: # Very horizontal (definitely lying)
                        fall_score += 0.25
                        fall_indicators += 1
                
                # STRICT REQUIREMENT: MUST HAVE AT LEAST 4 INDICATORS
                if fall_indicators < 4:
                    # Less than 4 indicators = likely normal movement (standing/sitting), not a fall
                    fall_score *= 0.20  # Heavily penalize
                elif fall_indicators == 4:
                    # 4 indicators = moderately confident
                    fall_score *= 0.75
                # 5+ indicators = high confidence, use full score
                
                is_falling = fall_score >= 0.90  # Higher threshold (was 0.75)
                
                if is_falling:
                    person_fall_states[person_id]['frames'] += 1
                else:
                    person_fall_states[person_id]['frames'] = 0
                
                fall_confirmed = person_fall_states[person_id]['frames'] >= 7
                
                if is_falling:
                    print(f"Frame {frame_count:3d} Person {person_id}: Score={fall_score:.2f} (FALLING), "
                          f"Frames={person_fall_states[person_id]['frames']}/7 "
                          f"HWR={HWR:.2f}, Torso={TorsoAngle:.1f}°, H={H:.2f}, Fall°={FallAngleD:.1f}°")
                
                if fall_confirmed:
                    print(f"           >>> FALL CONFIRMED AT FRAME {frame_count} FOR PERSON {person_id}")
                    fall_confirmed_frames.append((frame_count, person_id))
    
    cap.release()
    
    print(f"\n[SUMMARY]")
    print(f"Total frames scanned: {total_frames}")
    print(f"Total unique people detected: {next_person_id - 1}")
    print(f"Fall confirmations: {len(fall_confirmed_frames)}")
    if fall_confirmed_frames:
        print(f"Frames with confirmed falls:")
        for frame_num, person_id in fall_confirmed_frames:
            print(f"  - Frame {frame_num}: Person {person_id}")
    else:
        print("No confirmed falls detected in video")

if __name__ == "__main__":
    try:
        find_falls()
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
