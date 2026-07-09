#!/usr/bin/env python3
"""
Test script to simulate uploading a video and checking if detection works
"""
import os
import sys
import time
import requests
from pathlib import Path

# Config
BASE_URL = "http://localhost:5000"
VIDEO_PATH = "app/uploads/554478814_25532680373011618_6149359910214032878_n.mp4"

def test_upload_and_detect():
    """Test uploading a video and verifying detection"""
    
    # Check if video exists
    if not os.path.exists(VIDEO_PATH):
        print(f"ERROR: Video not found at {VIDEO_PATH}")
        return False
    
    print("\n=== Testing Video Upload and Detection ===")
    print(f"Video: {VIDEO_PATH}")
    
    # Try to get current cameras
    try:
        resp = requests.get(f"{BASE_URL}/api/cameras", timeout=5)
        if resp.status_code == 200:
            cameras_before = resp.json()
            print(f"\nCameras before upload: {len(cameras_before)}")
            for cam in cameras_before:
                print(f"  - {cam.get('name', 'Unknown')}: id={cam.get('id')}, status={cam.get('status')}, isLive={cam.get('isLive')}")
        else:
            print(f"ERROR: Failed to get cameras: {resp.status_code}")
            return False
    except Exception as e:
        print(f"ERROR: Could not connect to server: {e}")
        print(f"Make sure the server is running at {BASE_URL}")
        return False
    
    # Upload the video (NOTE: This will re-upload if run multiple times)
    print("\n--- Uploading video ---")
    try:
        with open(VIDEO_PATH, 'rb') as f:
            files = {
                'video_file': f,
            }
            data = {
                'name': 'Test Upload ' + time.strftime('%Y-%m-%d %H:%M:%S'),
            }
            resp = requests.post(f"{BASE_URL}/api/cameras/upload", files=files, data=data, timeout=30)
        
        if resp.status_code != 200:
            print(f"ERROR: Upload failed with status {resp.status_code}")
            print(f"Response: {resp.text}")
            return False
        
        result = resp.json()
        print(f"Upload successful: {result.get('message')}")
        camera_id = result.get('camera_id')
        print(f"New camera ID: {camera_id}")
    except Exception as e:
        print(f"ERROR: Upload failed: {e}")
        return False
    
    # Wait for processor to start
    print("\n--- Waiting for processor to initialize ---")
    time.sleep(2)
    
    # Check camera status over time
    print("\n--- Monitoring camera status ---")
    max_checks = 15
    for check_num in range(1, max_checks + 1):
        try:
            resp = requests.get(f"{BASE_URL}/api/cameras", timeout=5)
            if resp.status_code == 200:
                cameras = resp.json()
                new_cam = next((c for c in cameras if c.get('id') == camera_id), None)
                
                if new_cam:
                    print(f"\n[Check {check_num}] Camera {new_cam.get('name')}:")
                    print(f"  ID: {new_cam.get('id')}")
                    print(f"  Status: {new_cam.get('status')}")
                    print(f"  Color: {new_cam.get('color')}")
                    print(f"  isLive: {new_cam.get('isLive')}")
                    print(f"  FPS: {new_cam.get('fps', 0):.1f}")
                    print(f"  People detected: {new_cam.get('people_detected', 0)}")
                    print(f"  Confidence: {new_cam.get('confidence_score', 0):.3f}")
                else:
                    print(f"[Check {check_num}] Camera not found")
            else:
                print(f"[Check {check_num}] API error: {resp.status_code}")
        except Exception as e:
            print(f"[Check {check_num}] Error: {e}")
        
        time.sleep(2)
    
    print("\n=== Test Complete ===")
    return True

if __name__ == "__main__":
    test_upload_and_detect()
