import requests
import sys

# Test uploading a video file
url = "http://localhost:5000/api/cameras/upload"
test_file = "C:/Users/rosen/Downloads/1.mp4"

print(f"[TEST] Uploading test file: {test_file}")
print(f"[TEST] Target URL: {url}")

try:
    with open(test_file, 'rb') as f:
        files = {
            'video_file': f,
        }
        data = {
            'name': 'Test Upload Camera'
        }
        
        print("[TEST] Sending POST request...")
        response = requests.post(url, files=files, data=data, timeout=10)
        
        print(f"[TEST] Response Status: {response.status_code}")
        print(f"[TEST] Response Body: {response.text}")
        
except Exception as e:
    print(f"[TEST] ERROR: {e}")
    import traceback
    traceback.print_exc()
