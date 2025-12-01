#!/usr/bin/env python3
"""
Comprehensive test of skeleton drawing and LSTM integration
"""

import cv2
import numpy as np
import torch
from app.video_utils import detect_multiple_people, extract_8_kinematic_features
from app.skeleton_lstm import LSTMModel, SEQUENCE_LENGTH, FEATURE_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE

def test_skeleton_and_lstm():
    # Load LSTM model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[TEST] Device: {device}")
    
    MODEL_FILE = 'models/skeleton_lstm_pytorch_model.pth'
    LSTM_MODEL = None
    
    try:
        LSTM_MODEL = LSTMModel(FEATURE_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS)
        LSTM_MODEL.load_state_dict(torch.load(MODEL_FILE, map_location=device, weights_only=True))
        LSTM_MODEL.to(device)
        LSTM_MODEL.eval()
        print(f"[SUCCESS] Loaded LSTM Model")
        print(f"[CONFIG] Features: {FEATURE_SIZE}, Hidden: {HIDDEN_SIZE}, Output: {OUTPUT_SIZE}, Seq Len: {SEQUENCE_LENGTH}")
    except Exception as e:
        print(f"[WARNING] LSTM loading failed: {e}")
    
    # Test on video
    cap = cv2.VideoCapture('uploads/12.mp4')
    if not cap.isOpened():
        print("[ERROR] Cannot open video")
        return
    
    print("\n[TEST] Processing video...")
    frame_count = 0
    features_collected = 0
    
    while cap.isOpened() and frame_count < 100:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        frame_resized = cv2.resize(frame, (640, 480))
        
        # Detect people
        people = detect_multiple_people(frame_resized)
        
        if not people:
            continue
        
        print(f"\nFrame {frame_count}: Detected {len(people)} people")
        
        # Process first person
        person = people[0]
        landmarks = person['landmarks']
        
        # Extract features
        try:
            features_8 = extract_8_kinematic_features(landmarks)
            features_collected += 1
            
            print(f"  - Features extracted: HWR={features_8[0]:.3f}, TorsoAngle={features_8[1]:.1f}°, H={features_8[5]:.2f}")
            print(f"  - Landmarks type: {type(landmarks)}")
            print(f"  - Landmarks object has 'landmark' attr: {hasattr(landmarks, 'landmark')}")
            
            if hasattr(landmarks, 'landmark'):
                valid_lms = sum(1 for lm in landmarks.landmark if lm.visibility > 0.3)
                print(f"  - Valid landmarks (visibility > 0.3): {valid_lms}/33")
        
        except Exception as e:
            print(f"  - ERROR: Feature extraction failed: {e}")
    
    cap.release()
    
    print(f"\n[SUMMARY]")
    print(f"Frames processed: {frame_count}")
    print(f"Frames with people: {features_collected}")
    print(f"LSTM Model loaded: {'Yes' if LSTM_MODEL else 'No'}")

if __name__ == "__main__":
    test_skeleton_and_lstm()
