#!/usr/bin/env python3
"""Download YOLOv11n-pose model"""
from ultralytics import YOLO

print("[INFO] Downloading YOLOv11n-pose model...")
model = YOLO('yolov11n-pose')
print(f"[SUCCESS] Model downloaded and ready")
print(f"[INFO] Model will be saved as: runs/detect/yolov11n-pose.pt")
