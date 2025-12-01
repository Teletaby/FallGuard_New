#!/usr/bin/env python3
"""Test YOLOv8-Pose box inspection"""
import cv2
import numpy as np
from ultralytics import YOLO

# Load model
print("[INFO] Loading YOLOv8-Pose...")
try:
    model = YOLO('yolov8n-pose.pt')
    print("[SUCCESS] YOLOv8-Pose loaded")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    exit(1)

# Create dummy image (white background)
dummy_img = np.ones((480, 640, 3), dtype=np.uint8) * 255

# Run inference
print("[INFO] Running inference on dummy image...")
results = model(dummy_img, conf=0.5, verbose=False)

print(f"[DEBUG] Results length: {len(results)}")
for idx, result in enumerate(results):
    print(f"\n[DEBUG] Result {idx}:")
    print(f"  - has .boxes: {result.boxes is not None}")
    print(f"  - has .keypoints: {result.keypoints is not None}")
    
    if result.boxes is not None:
        print(f"  - len(boxes): {len(result.boxes)}")
        for i, box in enumerate(result.boxes):
            print(f"    Box {i}:")
            print(f"      - has .cls: {hasattr(box, 'cls')}")
            print(f"      - has .conf: {hasattr(box, 'conf')}")
            if hasattr(box, 'cls'):
                print(f"      - cls value: {box.cls}")
                print(f"      - cls type: {type(box.cls)}")
            if hasattr(box, 'conf'):
                print(f"      - conf value: {box.conf}")
    
    if result.keypoints is not None:
        print(f"  - keypoints type: {type(result.keypoints)}")
        print(f"  - keypoints.data shape: {result.keypoints.data.shape if hasattr(result.keypoints, 'data') else 'N/A'}")

print("\n[INFO] Test complete")
