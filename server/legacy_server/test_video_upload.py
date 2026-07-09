#!/usr/bin/env python3
"""
Test script to diagnose video upload and detection issues
"""
import os
import cv2
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

def test_video_file(video_path):
    """Test if a video file can be read and frames extracted"""
    print(f"\n=== Testing Video File: {video_path} ===")
    
    if not os.path.exists(video_path):
        print(f"ERROR: File does not exist: {video_path}")
        return False
    
    file_size = os.path.getsize(video_path) / (1024*1024)
    print(f"File size: {file_size:.2f} MB")
    print(f"File exists: Yes")
    print(f"File readable: {os.access(video_path, os.R_OK)}")
    
    # Try to open with OpenCV
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("ERROR: OpenCV cannot open this file!")
        return False
    
    print("OpenCV opened successfully")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"FPS: {fps}")
    print(f"Frame count: {frame_count}")
    print(f"Resolution: {width}x{height}")
    
    # Try to read first frame
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Cannot read first frame!")
        cap.release()
        return False
    
    print(f"First frame read successfully: shape={frame.shape}, dtype={frame.dtype}")
    
    # Try to read a few more frames
    for i in range(4):
        ret, frame = cap.read()
        if ret:
            print(f"Frame {i+1} read successfully")
        else:
            print(f"ERROR reading frame {i+1}")
            break
    
    cap.release()
    return True

def check_uploads_folder():
    """Check what's in the uploads folder"""
    uploads_path = os.path.join(os.path.dirname(__file__), "app", "uploads")
    print(f"\n=== Checking Uploads Folder ===")
    print(f"Path: {uploads_path}")
    print(f"Exists: {os.path.exists(uploads_path)}")
    
    if os.path.exists(uploads_path):
        files = os.listdir(uploads_path)
        print(f"Files in uploads folder ({len(files)}):")
        for f in files:
            full_path = os.path.join(uploads_path, f)
            size = os.path.getsize(full_path) / (1024*1024)
            print(f"  - {f} ({size:.2f} MB)")
            
            # If it looks like a video, test it
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')):
                test_video_file(full_path)

if __name__ == "__main__":
    check_uploads_folder()
    
    # Also check for test video files
    data_path = os.path.join(os.path.dirname(__file__), "data")
    if os.path.exists(data_path):
        print(f"\n=== Checking Data Folder ===")
        files = [f for f in os.listdir(data_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.csv'))]
        if files:
            print(f"Video/data files in {data_path}:")
            for f in files:
                full_path = os.path.join(data_path, f)
                size = os.path.getsize(full_path) / (1024*1024)
                print(f"  - {f} ({size:.2f} MB)")
                if f.lower().endswith(('.mp4', '.avi', '.mov')):
                    test_video_file(full_path)
