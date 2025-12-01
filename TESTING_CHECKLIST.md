# Multi-Person Detection Testing Checklist

## Pre-Deployment

### Code Quality
- [x] No syntax errors in modified files
- [x] All imports present (np, cv2, torch, mp)
- [x] Backward compatible with existing single-person logic
- [x] Added comprehensive logging statements

### Logic Validation
- [x] Three-stage detection properly implemented
- [x] IoU deduplication prevents duplicates
- [x] Person matching uses aspect ratio
- [x] Fall prediction independent per person
- [x] Timeout logic persists trackers correctly

---

## Testing with Your Uploaded Video

### Step 1: Visual Verification
```
1. Start the server:
   python run_server.py

2. Open browser: http://localhost:5000

3. Upload your 2-person video

4. Check live feed for:
   ☐ Person #1 visible with bounding box
   ☐ Person #2 visible with bounding box
   ☐ Both have different unique IDs
   ☐ Skeletons drawn for both
   ☐ IDs don't flicker/change
```

### Step 2: Frame Count Verification
```
In the browser console or logs, you should see:
[DETECTION] Found 2 person(s) in frame #50 - appears regularly
[DETECTION] Total people detected: 2 (deduplicated from 3-4 raw)
[TRACKING] Matched Person #1 (distance=XX, score=0.XX)
[TRACKING] Matched Person #2 (distance=XX, score=0.XX)
```

### Step 3: Fall Detection Testing
```
When either person falls:
☐ Person #1 or #2 shows "FALLING!" label
☐ Box turns red
☐ Telegram alert sent (check message)
☐ Website alert appears
☐ Correct person ID in alert message
```

### Step 4: Performance Metrics
```
Monitor (in status panel):
☐ FPS > 15 (ideally 20+)
☐ CPU < 60% for single camera
☐ Memory stable (no growth over 5 minutes)
☐ Latency < 500ms
```

---

## Regression Testing (Single Person Video)

### Existing Functionality
```
1. Test with old single-person video
   ☐ Still detects person correctly
   ☐ Shows Person #1
   ☐ Fall detection still works
   ☐ No false Person #2 ghost detection

2. Test with webcam (real-time)
   ☐ Live person detected
   ☐ Skeleton appears
   ☐ Smooth tracking (no ID flicker)
   ☐ Fall detection responsive
```

---

## Edge Cases to Test

### Scenario 1: People at Different Distances
```
Video: Person far away + Person close
Expected: Both detected
- [ ] Far person detected (may be smaller box)
- [ ] Near person detected (larger box)
- [ ] Size ratio matching doesn't merge them
```

### Scenario 2: People Partially Occluded
```
Video: One person partially behind another
Expected: Both still detected (MediaPipe extrapolates)
- [ ] Front person fully visible
- [ ] Back person partially visible
- [ ] Not merged into single detection
```

### Scenario 3: Fast Movement
```
Video: People moving quickly between frames
Expected: Tracking maintains ID
- [ ] Person #1 stays Person #1
- [ ] Person #2 stays Person #2
- [ ] No ID thrashing (flipping between IDs)
- [ ] Fall detection still responds
```

### Scenario 4: People Together (Proximity)
```
Video: Both people close together
Expected: Still separate detections
- [ ] Not merged into one person
- [ ] Aspect ratio matching helps here
- [ ] Both get independent fall scores
```

---

## Performance Stress Tests

### CPU Load
```
- [ ] Single 2-person video: 30-50% CPU
- [ ] Two simultaneous cameras: 50-75% CPU
- [ ] Three cameras: 75-90% CPU (acceptable)
- [ ] Four cameras: > 100% (needs optimization)
```

### Memory Stability
```
Monitor for 10 minutes:
- [ ] No memory creep (constant usage)
- [ ] No tracker array growing unbounded
- [ ] No duplicate frame buffer accumulation
- [ ] Garbage collection working normally
```

### Latency
```
From frame capture to alert:
- [ ] Detection latency: < 100ms
- [ ] Fall confirmation: 2-3 seconds (as configured)
- [ ] Telegram send: < 10 seconds
- [ ] Total alert latency: < 30 seconds
```

---

## Debug Commands

### Check Detection Logs
```powershell
# In PowerShell terminal, look for these lines:
[DETECTION] Full-frame detection at 
[DETECTION] LEFT split detection at 
[DETECTION] RIGHT split detection at 
[DETECTION] Total people detected: 2

# Count detections per minute:
Get-Content upload_logs.txt | Select-String "DETECTION" | Measure-Object
```

### Monitor Performance
```powershell
# CPU Usage
Get-Process | Where-Object {$_.ProcessName -match "python"} | 
  Select-Object ProcessName, @{Name="CPU%";Expression={$_.CPU}}

# Memory Usage
Get-Process | Where-Object {$_.ProcessName -match "python"} | 
  Select-Object ProcessName, @{Name="MemMB";Expression={[math]::Round($_.WorkingSet/1MB, 2)}}
```

### Test Fall Detection Directly
```python
# In Python console or test script:
from app.video_utils import detect_multiple_people
import cv2
import mediapipe as mp

cap = cv2.VideoCapture("path/to/2person_video.mp4")
mp_pose = mp.solutions.pose.Pose()

ret, frame = cap.read()
people = detect_multiple_people(frame, mp_pose)
print(f"Detected {len(people)} people")
for i, p in enumerate(people):
    print(f"  Person {i+1}: bbox={p['bbox']}, conf={p['confidence']:.2f}")
```

---

## Sign-Off Checklist

- [ ] All visual tests passed (both people detected)
- [ ] Performance metrics acceptable
- [ ] No regressions with single-person videos
- [ ] Edge cases handled correctly
- [ ] Logs are informative and match expected output
- [ ] Memory stable over extended runs
- [ ] Fall detection alerts working properly
- [ ] Telegram notifications sending correctly
- [ ] Dashboard displays both people with unique IDs
- [ ] Ready for production deployment

---

## Known Limitations (Document)

1. **Requires clear view of head/shoulders** - MediaPipe needs upper body
2. **Lighting matters** - Dark shadows reduce detection confidence
3. **Split detection has processing cost** - 20-30% more CPU
4. **Best with people at similar scales** - Aspect ratio matching helps
5. **No 3D depth** - Camera angle affects detection accuracy

---

## Rollback Plan (If Issues Arise)

If new detection causes problems:

1. Revert `app/video_utils.py` to use single full-frame detection
2. Set `pose_process_interval: 3` (original)
3. Set `person_timeout: 2.5` (original)
4. Update heuristics back to original thresholds
5. Or checkout: `git checkout main -- app/video_utils.py main.py`

---

## Contact Points for Debugging

If issues remain:
1. **Missing person**: Check MediaPipe visibility (needs 0.5+ confidence)
2. **ID flickering**: Person matching distance too small, increase threshold
3. **False falls**: Heuristic thresholds too aggressive, increase threshold
4. **CPU high**: Reduce `pose_process_interval` to 3, or 4
5. **Memory leak**: Check frame buffer size and person tracker cleanup

---

Generated: 2025-11-30
Last Updated: [Date of deployment]
