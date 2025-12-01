#!/usr/bin/env python3
"""
Quick test script to verify YOLOv11 detection on webcam or video file
"""
import cv2
import sys
from app.video_utils import detect_multiple_people, YOLO_MODEL

print("[INFO] Starting detection test...")
print(f"[INFO] YOLO_MODEL loaded: {YOLO_MODEL is not None}")

# Try webcam first
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot open webcam")
    sys.exit(1)

print("[INFO] Webcam opened, testing detection...")
print("[INFO] Press 'q' to quit. Showing first detection for 3 seconds...")

frame_count = 0
detection_found = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Cannot read frame")
        break
    
    frame_count += 1
    
    # Only process every 5 frames for speed
    if frame_count % 5 != 0:
        continue
    
    # Run detection
    people = detect_multiple_people(frame)
    
    if len(people) > 0 and not detection_found:
        detection_found = True
        print(f"\n[SUCCESS] Detected {len(people)} person(s)!")
        for i, person in enumerate(people):
            bbox = person['bbox']
            print(f"  Person {i+1}: bbox={bbox}, area={person['area']:.0f}, conf={person['confidence']:.2f}")
        
        # Draw bounding boxes
        display = frame.copy()
        for i, person in enumerate(people):
            x, y, w, h = person['bbox']
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(display, f"Person {i+1}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Show for a moment
        cv2.imshow("Detection Test", display)
        cv2.waitKey(3000)
        break
    
    if frame_count % 30 == 0:
        print(f"[FRAME] {frame_count}: {len(people)} people detected...")
    
    if frame_count > 300:  # Test 300 frames (10 seconds at 30 FPS)
        break

cap.release()
cv2.destroyAllWindows()

if detection_found:
    print("[SUCCESS] Detection test passed!")
else:
    print("[WARNING] No people detected in first 300 frames. Check:")
    print("  1. Is there a person in front of the camera?")
    print("  2. Is the lighting adequate?")
    print("  3. Try moving closer to the camera")
