import requests

try:
    # Test simple GET
    print("[TEST] Testing GET http://localhost:5001/api/test")
    r = requests.get("http://localhost:5001/api/test", timeout=5)
    print(f"GET Status: {r.status_code}, Body: {r.text}")
except Exception as e:
    print(f"GET Error: {e}")

try:
    # Test simple POST with form data
    print("[TEST] Testing POST http://localhost:5001/api/test")
    r = requests.post("http://localhost:5001/api/test", data={'test': 'data'}, timeout=5)
    print(f"POST Status: {r.status_code}, Body: {r.text}")
except Exception as e:
    print(f"POST Error: {e}")
