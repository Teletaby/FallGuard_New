#!/usr/bin/env python3
"""Debug script to trace skeleton drawing"""
import sys
import cv2
import time

# Add to path
sys.path.insert(0, '.')

from main import CameraProcessor, GLOBAL_SETTINGS, DEFAULT_FALL_THRESHOLD
import threading

# Create a camera processor for a test video
processor = CameraProcessor(
    camera_id=1,
    camera_name="Test_Camera",
    src="uploads/1.mp4"
)

# Run for a few frames with detailed logging
print("\n=== STARTING SKELETON DEBUG ===")
print(f"GLOBAL_SETTINGS: pose_process_interval={GLOBAL_SETTINGS.get('pose_process_interval')}")
print(f"Fall threshold: {GLOBAL_SETTINGS.get('fall_threshold')}")

# Override run method to add extra debugging
original_run = processor.run

def debug_run():
    print(f"[DEBUG] Starting run()")
    # Call original but with frame limit
    try:
        original_run()
    except KeyboardInterrupt:
        print("\n[DEBUG] Interrupted")
    except Exception as e:
        print(f"[DEBUG] Exception: {e}")
        import traceback
        traceback.print_exc()

# Run in thread
thread = threading.Thread(target=debug_run, daemon=True)
thread.start()

# Let it run for 15 seconds
time.sleep(15)
print("\n[DEBUG] Stopping after 15 seconds")

# Give time to cleanup
time.sleep(2)
print("=== DEBUG COMPLETE ===")
