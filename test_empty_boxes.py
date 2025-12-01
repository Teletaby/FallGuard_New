#!/usr/bin/env python3
"""Debug YOLOv8-Pose empty detection"""
from ultralytics import YOLO
import numpy as np

model = YOLO('yolov8n-pose.pt')
img = np.ones((480, 640, 3), dtype=np.uint8) * 255

results = model(img, conf=0.5, verbose=False)
result = results[0]

print("Result boxes:")
print(f"Type: {type(result.boxes)}")
print(f"Boxes is None: {result.boxes is None}")
print(f"Len check: {len(result.boxes) if result.boxes is not None else 'N/A'}")
print(f"Bool check: {bool(result.boxes) if result.boxes is not None else 'N/A'}")

# Try to enumerate
if result.boxes is not None:
    try:
        for i, box in enumerate(result.boxes):
            print(f"Box {i}: {box}")
    except Exception as e:
        print(f"Error enumerating: {e}")
        
# Check attributes directly
print(f"\nBoxes attributes: {dir(result.boxes) if result.boxes is not None else 'N/A'}")
