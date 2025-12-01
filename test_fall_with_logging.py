#!/usr/bin/env python3
"""
Test the actual fall detection with logging
"""

import cv2
import numpy as np
import torch
from collections import deque
import time
import os

# Import from app
from app.video_utils import detect_multiple_people, extract_8_kinematic_features
from app.skeleton_lstm import FEATURE_SIZE, SEQUENCE_LENGTH, LSTMModel, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE

def test_fall_detection_with_logging():
    """Test fall detection with detailed logging"""
    
    # Load model
    MODEL_FILE = 'models/skeleton_lstm_pytorch_model.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        LSTM_MODEL = LSTMModel(FEATURE_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS)
        if os.path.exists(MODEL_FILE):
            LSTM_MODEL.load_state_dict(torch.load(MODEL_FILE, map_location=device, weights_only=True))
            LSTM_MODEL.to(device)
            LSTM_MODEL.eval()
            print(f"[SUCCESS] Loaded LSTM Model")
        else:
            LSTM_MODEL = None
            print(f"[WARNING] LSTM Model not found")
    except Exception as e:
        print(f"[WARNING] LSTM error: {e}")
        LSTM_MODEL = None
    
    # Load video
    video_path = "uploads/12.mp4"
    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        return
    
    print(f"\n[INFO] Processing video: {video_path}\n")
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    sequence = deque([np.zeros(FEATURE_SIZE, dtype=np.float32) for _ in range(SEQUENCE_LENGTH)], maxlen=SEQUENCE_LENGTH)
    
    GLOBAL_SETTINGS = {
        "fall_threshold": 0.75,
    }
    
    while frame_count < 100:  # Process first 100 frames
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (640, 480))
        frame_count += 1
        
        # Detect people
        people = detect_multiple_people(frame, None, use_hog=False)
        
        if not people:
            print(f"Frame {frame_count}: No people detected")
            continue
        
        person = people[0]  # Take first person
        landmarks = person['landmarks']
        
        # Extract features
        features_8 = extract_8_kinematic_features(landmarks)
        feature_vec = np.zeros(FEATURE_SIZE, dtype=np.float32)
        feature_vec[:8] = features_8
        sequence.append(feature_vec)
        
        # Check features
        HWR = features_8[0]
        TorsoAngle = features_8[1]
        H = features_8[5]
        FallAngleD = features_8[6]
        
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
        lstm_fall = False
        lstm_prob = 0.0
        if LSTM_MODEL and len(sequence) >= SEQUENCE_LENGTH:
            input_data = np.array(list(sequence), dtype=np.float32)
            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = LSTM_MODEL(input_tensor)
                if len(output.shape) > 1:
                    lstm_prob = float(output[0, 0])
                else:
                    lstm_prob = float(output[0])
            
            lstm_prob = max(0.0, min(1.0, lstm_prob))
            lstm_fall = lstm_prob >= GLOBAL_SETTINGS['fall_threshold']
        
        # Overall prediction
        final_prob = max(fall_score, lstm_prob)
        is_fall = final_prob >= GLOBAL_SETTINGS['fall_threshold']
        
        if frame_count % 5 == 0 or is_fall:
            print(f"Frame {frame_count:3d}: HWR={HWR:.2f}, Torso={TorsoAngle:5.1f}°, H={H:.2f}, Fall°={FallAngleD:5.1f}°")
            print(f"           Heur={fall_score:.2f}, LSTM={lstm_prob:.2f}, Final={final_prob:.2f} → {'FALL' if is_fall else 'NORMAL'}")
            if is_fall:
                print("           ^^^ FALL DETECTED ^^^")
            print()
    
    cap.release()

if __name__ == "__main__":
    try:
        test_fall_detection_with_logging()
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
