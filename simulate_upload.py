#!/usr/bin/env python3
"""
Test to simulate what happens when you upload a video
"""

import cv2
import numpy as np
import os
import time

def simulate_upload_processing():
    """Simulate what happens during video upload"""
    
    video_path = "uploads/12.mp4"
    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        return
    
    print(f"\n[SIMULATION] Processing uploaded video: {video_path}")
    print("="*70)
    
    from app.video_utils import detect_multiple_people, extract_8_kinematic_features
    from collections import deque
    import torch
    from app.skeleton_lstm import FEATURE_SIZE, SEQUENCE_LENGTH, LSTMModel, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE
    
    # Load LSTM model (with error handling for shape mismatch)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_FILE = 'models/skeleton_lstm_pytorch_model.pth'
    LSTM_MODEL = None
    
    try:
        LSTM_MODEL = LSTMModel(FEATURE_SIZE, HIDDEN_SIZE, 1, NUM_LAYERS)  # Try OUTPUT_SIZE=1
        if os.path.exists(MODEL_FILE):
            LSTM_MODEL.load_state_dict(torch.load(MODEL_FILE, map_location=device, weights_only=True))
            LSTM_MODEL.to(device)
            LSTM_MODEL.eval()
            print("[SUCCESS] Loaded LSTM model (OUTPUT_SIZE=1)")
    except Exception as e:
        print(f"[WARNING] Could not load LSTM: {e}")
        LSTM_MODEL = None
    
    cap = cv2.VideoCapture(video_path)
    
    # Simulate camera processor behavior
    sequence = deque([np.zeros(FEATURE_SIZE, dtype=np.float32) for _ in range(SEQUENCE_LENGTH)], maxlen=SEQUENCE_LENGTH)
    person_fall_states = {}
    person_id = 1
    frame_count = 0
    consecutive_fall_frames = 0
    
    GLOBAL_SETTINGS = {"fall_threshold": 0.75}
    
    print("\n[INFO] Starting video processing simulation...")
    print("[INFO] Looking for 7+ consecutive frames to trigger fall alert\n")
    
    while frame_count < 50:  # Process first 50 frames
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (640, 480))
        frame_count += 1
        
        # Detect people
        people = detect_multiple_people(frame, None, use_hog=False)
        
        if not people:
            if frame_count % 5 == 0:
                print(f"Frame {frame_count:3d}: No detection")
            consecutive_fall_frames = 0
            continue
        
        person = people[0]
        landmarks = person['landmarks']
        
        # Extract features
        features_8 = extract_8_kinematic_features(landmarks)
        feature_vec = np.zeros(FEATURE_SIZE, dtype=np.float32)
        feature_vec[:8] = features_8
        sequence.append(feature_vec)
        
        HWR = features_8[0]
        TorsoAngle = features_8[1]
        H = features_8[5]
        FallAngleD = features_8[6]
        
        # Calculate heuristic fall score
        fall_score = 0.0
        
        if 0.0 < HWR < 0.45:
            fall_score += 0.35
            if HWR < 0.30:
                fall_score += 0.30
        
        if TorsoAngle > 65:
            fall_score += 0.30
            if TorsoAngle > 78:
                fall_score += 0.15
        
        if H > 0.72:
            fall_score += 0.08
            if H > 0.82:
                fall_score += 0.12
        
        if FallAngleD < 20:
            fall_score += 0.35
            if FallAngleD < 10:
                fall_score += 0.18
        
        is_fall_heuristic = fall_score >= GLOBAL_SETTINGS['fall_threshold']
        
        # LSTM prediction
        lstm_prob = 0.0
        is_fall_lstm = False
        
        if LSTM_MODEL and len(sequence) >= SEQUENCE_LENGTH:
            try:
                input_data = np.array(list(sequence), dtype=np.float32)
                input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = LSTM_MODEL(input_tensor)
                    if len(output.shape) > 1:
                        lstm_prob = float(output[0, 0])
                    else:
                        lstm_prob = float(output[0])
                
                lstm_prob = max(0.0, min(1.0, lstm_prob))
                is_fall_lstm = lstm_prob >= GLOBAL_SETTINGS['fall_threshold']
            except Exception as e:
                pass
        
        # Overall prediction
        is_falling = is_fall_heuristic or is_fall_lstm
        final_prob = max(fall_score, lstm_prob)
        
        # Fall detection confirmation logic
        if is_falling:
            consecutive_fall_frames += 1
        else:
            consecutive_fall_frames = 0
        
        fall_confirmed = consecutive_fall_frames >= 7
        
        status = "FALL CONFIRMED!!!" if fall_confirmed else ("fall detected (needs 7)" if is_falling else "NORMAL")
        
        if frame_count % 3 == 0 or is_falling or fall_confirmed:
            print(f"Frame {frame_count:3d}: HWR={HWR:.2f}, Torso={TorsoAngle:5.1f}°, H={H:.2f}, Fall°={FallAngleD:5.1f}°")
            print(f"           Heur={fall_score:.2f}(falling={is_fall_heuristic}), LSTM={lstm_prob:.2f}(falling={is_fall_lstm})")
            print(f"           Frames={consecutive_fall_frames}/7 → {status}")
            if is_falling or fall_confirmed:
                print()
    
    cap.release()
    
    print("\n" + "="*70)
    if consecutive_fall_frames >= 7:
        print("[RESULT] FALL WAS DETECTED AND ALERT SENT")
    else:
        print(f"[RESULT] No fall detected (max consecutive frames: {consecutive_fall_frames}/7)")

if __name__ == "__main__":
    try:
        simulate_upload_processing()
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
