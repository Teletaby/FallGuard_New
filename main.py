import os
import sys
import threading
import time
import json
import uuid
from flask import Flask, request, jsonify, session, send_from_directory, Response
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename 
import numpy as np
import cv2
from collections import deque
import requests
from datetime import datetime
import base64
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# --- IMPORT MODULES ---
import torch
from app.skeleton_lstm import LSTMModel, SEQUENCE_LENGTH, FEATURE_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE
from app.video_utils import extract_55_features, predict_torch, detect_multiple_people, draw_skeleton_yolo, extract_8_kinematic_features

# --- Global Settings ---
DEFAULT_FALL_THRESHOLD = 0.75
INTERNAL_FPS = 30
DEFAULT_FALL_DELAY_SECONDS = 2
DEFAULT_ALERT_COOLDOWN_SECONDS = 60

GLOBAL_SETTINGS = {
    "fall_threshold": 0.95,  # Very high threshold = fewer false positives
    "fall_delay_seconds": 3,  # Longer delay = more confirmation required
    "alert_cooldown_seconds": 60,
    "privacy_mode": "full_video",
    "pre_fall_buffer_seconds": 5,
    "pose_process_interval": 2,  # Process every 2 frames for better FPS
    "use_yolov11": True
}

# Telegram Settings
TELEGRAM_BOT_TOKEN = "8204879198:AAErRTPpGXDZGsXO7ZoF9VtTWbDJB9isxzA"
TELEGRAM_SUBSCRIBERS = []
TELEGRAM_SUBSCRIBERS_FILE = "data/telegram_subscribers.json"
TELEGRAM_BLOCKED_LIST = []
TELEGRAM_BLOCKED_LIST_FILE = "data/telegram_blocked_list.json"
telegram_lock = threading.Lock()

# Website Alerts - FIXED: Initialize properly
WEBSITE_ALERTS = {}
website_alerts_lock = threading.Lock()

# Alert tracking to prevent duplicates
ALERT_HISTORY = {}
ALERT_COOLDOWN = 60  # seconds

# Admin Password
ADMIN_PASSWORD = "admin"

def ensure_camera_status_exists(camera_id, name):
    """Ensure camera has status entry from the start"""
    with camera_lock:
        if camera_id not in CAMERA_STATUS:
            CAMERA_STATUS[camera_id] = {
                "status": "Initializing",
                "color": "gray",
                "isLive": False,
                "name": name,
                "source": "",
                "confidence_score": 0.0,
                "model_threshold": GLOBAL_SETTINGS['fall_threshold'],
                "fps": 0.0
            }

def save_subscribers_to_file():
    """Save subscribers to persistent JSON file"""
    try:
        os.makedirs(os.path.dirname(TELEGRAM_SUBSCRIBERS_FILE), exist_ok=True)
        with open(TELEGRAM_SUBSCRIBERS_FILE, 'w') as f:
            json.dump(TELEGRAM_SUBSCRIBERS, f, indent=2)
    except Exception as e:
        print(f"[ERROR] Failed to save subscribers to file: {e}")

def load_subscribers_from_file():
    """Load subscribers from persistent JSON file"""
    global TELEGRAM_SUBSCRIBERS
    try:
        if os.path.exists(TELEGRAM_SUBSCRIBERS_FILE):
            with open(TELEGRAM_SUBSCRIBERS_FILE, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    TELEGRAM_SUBSCRIBERS = data
                    print(f"[TELEGRAM] Loaded {len(TELEGRAM_SUBSCRIBERS)} subscriber(s) from file")
                    return
    except Exception as e:
        print(f"[ERROR] Failed to load subscribers from file: {e}")
    
    TELEGRAM_SUBSCRIBERS = []

def load_blocked_list_from_file():
    """Load blocked chat IDs from persistent JSON file"""
    global TELEGRAM_BLOCKED_LIST
    try:
        if os.path.exists(TELEGRAM_BLOCKED_LIST_FILE):
            with open(TELEGRAM_BLOCKED_LIST_FILE, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    TELEGRAM_BLOCKED_LIST = data
                    print(f"[TELEGRAM] Loaded {len(TELEGRAM_BLOCKED_LIST)} blocked user(s) from file")
                    return
    except Exception as e:
        print(f"[ERROR] Failed to load blocked list from file: {e}")
    
    TELEGRAM_BLOCKED_LIST = []

def save_blocked_list_to_file():
    """Save blocked chat IDs to persistent JSON file"""
    try:
        os.makedirs(os.path.dirname(TELEGRAM_BLOCKED_LIST_FILE), exist_ok=True)
        with open(TELEGRAM_BLOCKED_LIST_FILE, 'w') as f:
            json.dump(TELEGRAM_BLOCKED_LIST, f, indent=2)
    except Exception as e:
        print(f"[ERROR] Failed to save blocked list to file: {e}")

# --- Model Loading ---
MODEL_FILE = 'models/skeleton_lstm_pytorch_model.pth'
LSTM_MODEL = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Initializing PyTorch. Using device: {device}")

try:
    try:
        LSTM_MODEL = LSTMModel(FEATURE_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS)
        
        if os.path.exists(MODEL_FILE):
            LSTM_MODEL.load_state_dict(torch.load(MODEL_FILE, map_location=device, weights_only=True))
            LSTM_MODEL.to(device)
            LSTM_MODEL.eval()
            print(f"[SUCCESS] Loaded LSTM Model from {MODEL_FILE}")
            print(f"           Features: {FEATURE_SIZE}, Hidden: {HIDDEN_SIZE}, Output: {OUTPUT_SIZE}, Layers: {NUM_LAYERS}")
        else:
            print(f"[WARNING] Model file not found: {MODEL_FILE}")
            LSTM_MODEL = None
    except RuntimeError as e:
        if "size mismatch" in str(e):
            print(f"[INFO] Model shape mismatch - trying with output_size=1 (legacy format)")
            # The saved model was trained with output_size=1 (binary classification)
            # Create model with correct shape and load
            LSTM_MODEL = LSTMModel(FEATURE_SIZE, HIDDEN_SIZE, 1, NUM_LAYERS)
            
            if os.path.exists(MODEL_FILE):
                try:
                    LSTM_MODEL.load_state_dict(torch.load(MODEL_FILE, map_location=device, weights_only=True))
                    LSTM_MODEL.to(device)
                    LSTM_MODEL.eval()
                    print(f"[SUCCESS] Loaded LSTM Model (legacy format with output_size=1)")
                except Exception as e2:
                    print(f"[ERROR] Failed to load with output_size=1: {e2}")
                    LSTM_MODEL = None
            else:
                LSTM_MODEL = None
        else:
            raise
except Exception as e:
    print(f"[ERROR] Failed to load LSTM model: {e}") 
    print(f"[INFO] Using HEURISTIC ONLY for fall detection (LSTM disabled)")
    LSTM_MODEL = None

# YOLOv11 Setup - automatically initialized when imported
print("[INFO] YOLOv11n-pose will be used for all person detection")

# Fall Timer Logic
class FallTimer:
    def __init__(self, threshold_frames=5):
        self.threshold = threshold_frames
        self.counter = 0
        self.last_fall_time = 0
    
    def update(self, is_falling):
        current_time = time.time()
        if is_falling:
            self.counter += 1
            self.last_fall_time = current_time
        else:
            if current_time - self.last_fall_time > 1.0:
                self.counter = 0
        return self.counter >= self.threshold

# --- GLOBAL CAMERA MANAGEMENT ---
CAMERA_DEFINITIONS = {}
CAMERA_STATUS = {}
shared_frames = {}
camera_lock = threading.Lock()

# Frame buffer for pre-fall recording
FRAME_BUFFERS = {}
BUFFER_SIZE = 75  # 2.5 seconds at 30 FPS (reduced for better memory usage)

# Incident Reports Storage
INCIDENT_REPORTS = []
INCIDENT_REPORTS_FILE = "data/incident_reports.json"
SNAPSHOTS_DIR = "data/snapshots"

def save_incident_reports():
    """Save incident reports to persistent JSON file"""
    try:
        os.makedirs(os.path.dirname(INCIDENT_REPORTS_FILE), exist_ok=True)
        with open(INCIDENT_REPORTS_FILE, 'w') as f:
            json.dump(INCIDENT_REPORTS, f, indent=2)
    except Exception as e:
        print(f"[ERROR] Failed to save incident reports: {e}")

def load_incident_reports():
    """Load incident reports from persistent JSON file"""
    global INCIDENT_REPORTS
    try:
        if os.path.exists(INCIDENT_REPORTS_FILE):
            with open(INCIDENT_REPORTS_FILE, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    INCIDENT_REPORTS = data
                    print(f"[INFO] Loaded {len(INCIDENT_REPORTS)} incident report(s)")
                    return
    except Exception as e:
        print(f"[ERROR] Failed to load incident reports: {e}")
    
    INCIDENT_REPORTS = []

# Flask app 
app = Flask(__name__, static_folder='app', static_url_path='')
app.secret_key = os.environ.get("FALLGUARD_SECRET", "fallguard_secret_key_2024")
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(SNAPSHOTS_DIR):
    os.makedirs(SNAPSHOTS_DIR)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

# Telegram Functions
def send_telegram_message(chat_id, text):
    """Send text message to Telegram chat"""
    if not TELEGRAM_BOT_TOKEN:
        return False
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get('ok', False)
        return False
    except Exception as e:
        print(f"[ERROR] Failed to send Telegram message: {e}")
        return False

def send_telegram_video(chat_id, video_bytes, caption, filename="fall_incident.mp4"):
    """Send video to Telegram chat"""
    if not TELEGRAM_BOT_TOKEN:
        return False
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendVideo"
    
    try:
        files = {'video': (filename, video_bytes, 'video/mp4')}
        data = {
            'chat_id': chat_id,
            'caption': caption,
            'parse_mode': 'HTML'
        }
        response = requests.post(url, files=files, data=data, timeout=60)
        if response.status_code == 200:
            resp_data = response.json()
            return resp_data.get('ok', False)
        return False
    except Exception as e:
        print(f"[ERROR] Failed to send Telegram video: {e}")
        return False

def send_telegram_photo(chat_id, photo_bytes, caption):
    """Send photo to Telegram chat"""
    if not TELEGRAM_BOT_TOKEN:
        return False
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    
    try:
        files = {'photo': ('fall_detection.jpg', photo_bytes, 'image/jpeg')}
        data = {
            'chat_id': chat_id,
            'caption': caption,
            'parse_mode': 'HTML'
        }
        response = requests.post(url, files=files, data=data, timeout=30)
        if response.status_code == 200:
            resp_data = response.json()
            return resp_data.get('ok', False)
        return False
    except Exception as e:
        print(f"[ERROR] Failed to send Telegram photo: {e}")
        return False

def get_bot_info():
    """Get Telegram bot information"""
    if not TELEGRAM_BOT_TOKEN:
        return None
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getMe"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json().get('result', {})
    except Exception as e:
        print(f"[ERROR] Failed to get bot info: {e}")
    return None

def load_previous_subscribers():
    """Load all previous subscribers from Telegram message history"""
    global TELEGRAM_SUBSCRIBERS
    
    if not TELEGRAM_BOT_TOKEN:
        return
    
    try:
        print("[TELEGRAM] Loading previous subscribers from message history...")
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
        params = {'offset': 0, 'limit': 1000, 'timeout': 10}
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            updates = data.get('result', [])
            loaded_ids = set()
            
            for update in reversed(updates):
                message = update.get('message', {})
                text = message.get('text', '')
                chat_id = message.get('chat', {}).get('id')
                username = message.get('chat', {}).get('username', '')
                first_name = message.get('chat', {}).get('first_name', '')
                last_name = message.get('chat', {}).get('last_name', '')
                
                if text.startswith('/start') and chat_id and chat_id not in loaded_ids:
                    with telegram_lock:
                        if not any(sub['chat_id'] == str(chat_id) for sub in TELEGRAM_SUBSCRIBERS):
                            name = f"{first_name} {last_name}".strip() or "User"
                            TELEGRAM_SUBSCRIBERS.append({
                                'chat_id': str(chat_id),
                                'name': name,
                                'username': username
                            })
                            loaded_ids.add(chat_id)
                            print(f"[TELEGRAM] Loaded previous subscriber: {name} (ID: {chat_id})")
            
            if loaded_ids:
                print(f"[TELEGRAM] Successfully loaded {len(loaded_ids)} previous subscriber(s)")
                save_subscribers_to_file()
            else:
                print("[TELEGRAM] No previous subscribers found in message history")
    except Exception as e:
        print(f"[ERROR] Failed to load previous subscribers: {e}")

def check_telegram_updates():
    """Background thread to check for new Telegram subscribers"""
    last_update_id = 0
    
    load_previous_subscribers()
    
    while True:
        if not TELEGRAM_BOT_TOKEN:
            time.sleep(5)
            continue
        
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
            params = {'offset': last_update_id + 1, 'timeout': 30}
            response = requests.get(url, params=params, timeout=35)
            
            if response.status_code == 200:
                data = response.json()
                updates = data.get('result', [])
                
                for update in updates:
                    last_update_id = update['update_id']
                    message = update.get('message', {})
                    text = message.get('text', '')
                    chat_id = message.get('chat', {}).get('id')
                    username = message.get('chat', {}).get('username', '')
                    first_name = message.get('chat', {}).get('first_name', '')
                    last_name = message.get('chat', {}).get('last_name', '')
                    
                    if text.startswith('/start') and chat_id:
                        with telegram_lock:
                            if str(chat_id) in TELEGRAM_BLOCKED_LIST:
                                send_telegram_message(
                                    chat_id,
                                    "❌ You have been removed from FallGuard alerts."
                                )
                                print(f"[TELEGRAM] Blocked user tried /start: ID {chat_id}")
                                continue
                            
                            if not any(sub['chat_id'] == str(chat_id) for sub in TELEGRAM_SUBSCRIBERS):
                                name = f"{first_name} {last_name}".strip() or "User"
                                TELEGRAM_SUBSCRIBERS.append({
                                    'chat_id': str(chat_id),
                                    'name': name,
                                    'username': username
                                })
                                save_subscribers_to_file()
                                send_telegram_message(
                                    chat_id,
                                    f"✅ <b>Welcome to FallGuard!</b>\n\n"
                                    f"You will now receive fall detection alerts.\n"
                                    f"Your Chat ID: <code>{chat_id}</code>"
                                )
                                print(f"[TELEGRAM] New subscriber: {name} (ID: {chat_id})")
                            else:
                                send_telegram_message(
                                    chat_id,
                                    "ℹ️ You are already subscribed to fall alerts."
                                )
        except Exception as e:
            print(f"[ERROR] Telegram update check failed: {e}")
        
        time.sleep(1)

# Start Telegram bot listener
telegram_thread = threading.Thread(target=check_telegram_updates, daemon=True)
telegram_thread.start()

# Privacy Mode Functions
def apply_privacy_mode(frame, mode):
    """Apply privacy mode to frame"""
    if mode == "full_video":
        return frame
    elif mode == "skeleton_only":
        # Return black frame with just skeleton (will be drawn later)
        black_frame = np.zeros_like(frame)
        return black_frame
    elif mode == "blurred":
        # Apply heavy blur to protect identity
        blurred = cv2.GaussianBlur(frame, (51, 51), 30)
        return blurred
    elif mode == "alerts_only":
        # Return completely black frame
        return np.zeros_like(frame)
    else:
        return frame

# Video Processing Functions
def frames_to_video(frames, fps=30):
    """Convert list of frames to MP4 video bytes"""
    if not frames:
        return None
    
    height, width = frames[0].shape[:2]
    
    # Use temporary file
    temp_filename = f"temp_{uuid.uuid4().hex}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_filename, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    out.release()
    
    # Read the file back into memory
    with open(temp_filename, 'rb') as f:
        video_bytes = f.read()
    
    # Clean up temporary file
    try:
        os.remove(temp_filename)
    except:
        pass
    
    return video_bytes

# --- Enhanced Camera Processor ---
class CameraProcessor(threading.Thread):
    def __init__(self, camera_id, src, name, sequence_length=SEQUENCE_LENGTH, device=None):
        super().__init__(daemon=True)
        self.camera_id = camera_id
        self.src = src 
        self.name = name
        self.cap = None
        self.is_running = False
        self.device = device if device is not None else torch.device('cpu')
        
        self.fall_timer = FallTimer(threshold_frames=1) 
        self.last_alert_time = 0
        self.last_fall_detection_time = 0
        self.consecutive_fall_frames = 0
        self.alert_sent_for_current_fall = False  # Track if alert already sent for current fall
        self.current_fall_id = None  # Track current fall session
        
        # Multi-person tracking
        self.people_trackers = {}  # Dictionary to store tracker for each person (person_id -> tracker_state)
        self.next_person_id = 1
        # How many seconds without detection before a tracker is removed
        self.person_timeout = 2.5  # seconds
        self.person_pose_sequences = {}  # Track pose sequences per person
        self.person_fall_states = {}  # Track fall state per person
        
        # YOLOv11 is initialized globally in video_utils.py
        self.mp_pose_instance = None
        
        self.sequence_length = sequence_length
        self.pose_sequence = deque([np.zeros(FEATURE_SIZE, dtype=np.float32) for _ in range(sequence_length)], 
                                     maxlen=sequence_length) 
        
        self.latest_pose_results = None
        self.latest_fall_prob = 0.0
        self.latest_features = None
        
        self.frame_count = 0
        self.last_fps_update = time.time()
        self.current_fps = 0
        self.processing_time = 0
        
        # Initialize frame buffer for pre-fall recording
        FRAME_BUFFERS[self.camera_id] = deque(maxlen=BUFFER_SIZE)
        
        # Ensure status exists immediately
        ensure_camera_status_exists(self.camera_id, self.name)
        
        self._init_shared_frame()

    def _init_shared_frame(self):
        placeholder = 100 * np.ones((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Initializing...", (180, 220), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(placeholder, self.name, (200, 260), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
        
        shared_frames[self.camera_id] = {
            "frame": placeholder,
            "lock": threading.Lock()
        }

    def _match_person(self, bbox, threshold=600):
        """
        Match a detected person to an existing tracker using multi-criteria matching.
        Returns person_id if match found, None otherwise.
        """
        x, y, w, h = bbox
        bbox_center = (x + w/2, y + h/2)
        bbox_size = w * h
        bbox_aspect = h / w if w > 0 else 0
        
        best_match = None
        best_score = 0.0
        
        for person_id, tracker in self.people_trackers.items():
            tracker_center = tracker['center']
            tracker_bbox = tracker['bbox']
            tracker_size = tracker_bbox[2] * tracker_bbox[3]
            tracker_aspect = tracker_bbox[3] / tracker_bbox[2] if tracker_bbox[2] > 0 else 0
            
            # 1. Calculate spatial distance between centers
            distance = np.sqrt((bbox_center[0] - tracker_center[0])**2 + (bbox_center[1] - tracker_center[1])**2)
            
            # 2. Size consistency - allow wider variation for people at different distances
            size_ratio = min(bbox_size, tracker_size) / max(bbox_size, tracker_size) if max(bbox_size, tracker_size) > 0 else 0
            
            # 3. Aspect ratio consistency
            aspect_ratio = min(bbox_aspect, tracker_aspect) / max(bbox_aspect, tracker_aspect) if max(bbox_aspect, tracker_aspect) > 0 else 0
            
            # 4. Distance-based confidence with WIDER matching range for multi-person
            position_confidence = np.exp(-distance / 300.0) if distance < 800 else 0.0
            
            # 5. Size consistency score: More lenient for people at different distances
            if size_ratio >= 0.15:
                size_confidence = size_ratio
            else:
                size_confidence = 0.0
            
            # 6. Aspect ratio consistency: Very lenient
            if aspect_ratio >= 0.40:
                aspect_confidence = aspect_ratio
            else:
                aspect_confidence = 0.0
            
            # 7. Combined score with adjusted weights for multi-person detection
            combined_score = (position_confidence * 0.75) + (size_confidence * 0.15) + (aspect_confidence * 0.10)
            
            # Match if close enough and score is good
            if distance < threshold and combined_score > best_score:
                best_score = combined_score
                best_match = person_id
        
        return best_match

    def update_fall_timer_threshold(self):
        delay_seconds = GLOBAL_SETTINGS['fall_delay_seconds']
        frame_threshold = max(3, round(delay_seconds * INTERNAL_FPS))
        self.fall_timer = FallTimer(threshold_frames=frame_threshold)

    def update_camera_status(self, status, color, last_alert=None, is_live=True, person_count=0):
        with camera_lock:
            if self.camera_id not in CAMERA_STATUS:
                CAMERA_STATUS[self.camera_id] = {}
            
            CAMERA_STATUS[self.camera_id] = {
                "status": status,
                "color": color,
                "isLive": is_live and self.is_running,
                "name": self.name,
                "source": str(self.src),
                "confidence_score": float(self.latest_fall_prob),
                "model_threshold": GLOBAL_SETTINGS['fall_threshold'],
                "fps": float(self.current_fps),
                "people_detected": person_count
            }
            
            if last_alert:
                CAMERA_STATUS[self.camera_id]["lastAlert"] = time.ctime(last_alert)
            
            # Sync with camera definitions
            if self.camera_id in CAMERA_DEFINITIONS:
                CAMERA_DEFINITIONS[self.camera_id]['isLive'] = is_live and self.is_running

    def trigger_website_alert(self, fall_prob, person_id=None):
        """
        Trigger website alert for this camera with enhanced multi-person support.
        """
        alert_id = f"{self.camera_id}_person{person_id}_{int(time.time() * 1000)}" if person_id else f"{self.camera_id}_{int(time.time() * 1000)}"
        
        # Build detailed camera name with person info
        camera_display_name = f"{self.name}"
        if person_id:
            camera_display_name = f"{self.name} (Person #{person_id})"
        
        # Create alert with all necessary information
        alert_data = {
            "camera_id": self.camera_id,
            "camera_name": camera_display_name,
            "confidence": float(fall_prob),
            "timestamp": time.time(),
            "alert_id": alert_id,
            "person_id": person_id,
            "fall_severity": "HIGH" if fall_prob > 0.85 else "MEDIUM" if fall_prob > 0.65 else "LOW"
        }
        
        # Store alert with lock for thread safety
        with website_alerts_lock:
            # For multiple people, create a separate key per person
            key = f"{self.camera_id}_person{person_id}" if person_id else self.camera_id
            WEBSITE_ALERTS[key] = alert_data
        
        print(f"[ALERT] *** Website alert for {camera_display_name}: {fall_prob*100:.1f}% confidence (ID: {person_id})")

    def send_fall_alert_with_video(self, current_frame, fall_prob, person_id=None):
        """Send fall alert with pre-fall video buffer - LIMITED TO ONE PER FALL - SUPPORTS MULTIPLE PEOPLE"""
        current_time = time.time()
        cooldown = GLOBAL_SETTINGS['alert_cooldown_seconds']
        
        # Create unique key for each person
        alert_key = f"{self.camera_id}_person{person_id}_{int(current_time // 30)}" if person_id else f"{self.camera_id}_{int(current_time // 30)}"
        
        # Check cooldown and prevent duplicate alerts per person
        if alert_key in ALERT_HISTORY and current_time - ALERT_HISTORY[alert_key] < cooldown:
            return
        
        ALERT_HISTORY[alert_key] = current_time
        
        # Trigger website alert
        self.trigger_website_alert(fall_prob, person_id)
        
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_SUBSCRIBERS:
            print(f"[TELEGRAM] No bot token or subscribers configured")
            return
        
        # Get pre-fall frames from buffer
        pre_fall_frames = list(FRAME_BUFFERS[self.camera_id])
        if current_frame is not None:
            pre_fall_frames.append(current_frame)
        
        if not pre_fall_frames:
            return
        
        # Convert frames to video in a separate thread to avoid blocking
        def send_alert_async():
            try:
                # Convert frames to video
                video_bytes = frames_to_video(pre_fall_frames, fps=15)
                
                if not video_bytes:
                    return
                
                # Prepare alert message
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                confidence = fall_prob * 100
                
                person_info = f"Person #{person_id}: " if person_id else ""
                
                caption = (
                    f"*** FALL DETECTED - IMMEDIATE ATTENTION REQUIRED ***\n\n"
                    f"LOCATION: {self.name}\n"
                    f"PERSON: {person_info if person_id else 'Detected Person'}\n"
                    f"TIME: {timestamp}\n"
                    f"CONFIDENCE: {confidence:.1f}%\n\n"
                    f"EMERGENCY: Person may be injured!\n"
                    f"PLEASE CHECK IMMEDIATELY AND CALL FOR HELP IF NEEDED\n"
                    f"Video showing {len(pre_fall_frames)//15}s before fall"
                )
                
                # Send to all subscribers
                with telegram_lock:
                    for subscriber in TELEGRAM_SUBSCRIBERS:
                        try:
                            if len(video_bytes) < 50 * 1024 * 1024:
                                success = send_telegram_video(subscriber['chat_id'], video_bytes, caption, "fall_incident.mp4")
                            else:
                                # Fall back to photo if video too large
                                ret, jpeg = cv2.imencode('.jpg', current_frame)
                                photo_bytes = jpeg.tobytes()
                                success = send_telegram_photo(subscriber['chat_id'], photo_bytes, caption + "\n\n(Video too large, sending snapshot)")
                            
                            if success:
                                print(f"[TELEGRAM] Alert sent to {subscriber['name']} for {self.name} Person {person_id if person_id else '?'}")
                            else:
                                print(f"[TELEGRAM] Failed to send alert to {subscriber['name']}")
                        except Exception as e:
                            print(f"[ERROR] Failed to send alert to {subscriber['name']}: {e}")
            except Exception as e:
                print(f"[ERROR] Error in async alert sending: {e}")
        
        # Start alert in background thread
        alert_thread = threading.Thread(target=send_alert_async, daemon=True)
        alert_thread.start()

    def _predict_fall_for_person(self, person_id, feature_vec):
        """Predict fall for a specific person using EXTREMELY CONSERVATIVE thresholds"""
        current_threshold = GLOBAL_SETTINGS['fall_threshold']
        fall_probability = 0.0
        heuristic_prob = 0.0  # Initialize to avoid UnboundLocalError
        
        if feature_vec is not None and np.any(feature_vec != 0):
            HWR = feature_vec[0]        # Height-to-width ratio
            TorsoAngle = feature_vec[1] # Torso angle from vertical
            D = feature_vec[2]          # Head to hip vertical distance
            H = feature_vec[5]          # Hip height in frame
            FallAngleD = feature_vec[6] # Body angle from horizontal
            
            # CRITICAL: Explicitly reject standing poses - NEVER flag as fall
            # Standing: HWR > 1.5, TorsoAngle < 30°, H < 0.6
            if HWR > 1.5 and TorsoAngle < 30 and H < 0.6:
                # This is definitely a standing person - reject immediately
                heuristic_prob = 0.0
                return (False, 0.0)
            
            fall_score = 0.0
            fall_indicators = 0
            
            # EXTREMELY CONSERVATIVE THRESHOLDS - Only detect clear falls, NEVER standing
            # Standing: HWR ~2.5-4.0, Sitting: HWR ~1.0-2.0, Lying: HWR ~0.2-0.5
            # Only trigger if HWR is VERY low (definitely lying, not sitting)
            if 0.0 < HWR < 0.25:  # Only EXTREMELY flat poses (definitely lying)
                fall_score += 0.20
                fall_indicators += 1
                if HWR < 0.15:    # Extremely flat (definitely lying)
                    fall_score += 0.25
                    fall_indicators += 1
            
            # TorsoAngle: standing ~5-15°, sitting ~30-50°, lying >75°
            # Only trigger if VERY tilted (definitely not standing)
            if TorsoAngle > 85:   # Only EXTREMELY tilted poses (>85°)
                fall_score += 0.20
                fall_indicators += 1
                if TorsoAngle > 90: # Severely tilted (>90°)
                    fall_score += 0.20
                    fall_indicators += 1
            
            # H (hip height): standing ~0.3-0.4, sitting ~0.5-0.6, lying ~0.8+
            # Only trigger if hips are VERY low (definitely not standing)
            if H > 0.88:          # Only EXTREMELY low positions (near bottom)
                fall_score += 0.15
                fall_indicators += 1
                if H > 0.92:      # Extremely low
                    fall_score += 0.15
                    fall_indicators += 1
            
            # FallAngleD: standing ~75-85°, lying <15°
            # Only trigger if VERY horizontal (definitely lying)
            if FallAngleD < 12:   # Only EXTREMELY horizontal (lying flat)
                fall_score += 0.20
                fall_indicators += 1
                if FallAngleD < 5: # Very horizontal (definitely lying)
                    fall_score += 0.20
                    fall_indicators += 1
            
            # D (head-hip distance): standing has large distance, lying has small
            # Only trigger if VERY compressed (definitely not standing)
            if abs(D) < 0.04:     # Only EXTREMELY compressed (head very near hips)
                fall_score += 0.10
                fall_indicators += 1
            
            # CRITICAL: MUST HAVE AT LEAST 5 INDICATORS (was 4)
            # Standing people typically have 0-2 indicators, this ensures we never flag them
            if fall_indicators < 5:
                # Less than 5 indicators = definitely NOT a fall (standing/sitting)
                fall_score = 0.0  # Completely reject
            elif fall_indicators == 5:
                # 5 indicators = moderately confident
                fall_score *= 0.70
            # 6+ indicators = high confidence, use full score
            
            heuristic_prob = min(fall_score, 0.99)
        
        # LSTM prediction (if model is loaded and we have enough frames)
        lstm_prob = 0.0
        if LSTM_MODEL is not None and hasattr(self, 'person_pose_sequences') and person_id in self.person_pose_sequences:
            pose_seq = self.person_pose_sequences[person_id]
            if len(pose_seq) >= SEQUENCE_LENGTH:
                try:
                    # Get last SEQUENCE_LENGTH frames and convert to numpy array first
                    frames = list(pose_seq)[-SEQUENCE_LENGTH:]
                    frames_array = np.array(frames, dtype=np.float32)
                    input_tensor = torch.tensor(frames_array, dtype=torch.float32).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        output = LSTM_MODEL(input_tensor)
                        # Handle different output shapes
                        if len(output.shape) > 1:
                            lstm_prob = float(output[0, 0])
                        else:
                            lstm_prob = float(output[0])
                    
                    # Clamp probability to [0, 1]
                    lstm_prob = max(0.0, min(1.0, lstm_prob))
                
                except Exception as e:
                    print(f"[LSTM ERROR] Person {person_id}: {type(e).__name__}: {e}")
                    lstm_prob = 0.0
        
        # Combine predictions conservatively - but heuristic takes priority if it says "not falling"
        # If heuristic says 0.0 (standing), don't let LSTM override it
        if heuristic_prob == 0.0:
            fall_probability = 0.0  # Standing person - never flag as fall
        else:
            fall_probability = max(lstm_prob, heuristic_prob)
        
        return (fall_probability >= current_threshold), fall_probability

    def run(self):
        self.update_fall_timer_threshold()
        self.update_camera_status("Starting...", "gray", is_live=True)
        
        print(f"[{self.name}] Opening video source: {self.src}")
        
        max_retries = 3
        for attempt in range(max_retries):
            print(f"[{self.name}] Attempt {attempt + 1}: Opening source: {self.src} (type: {type(self.src).__name__})")
            self.cap = cv2.VideoCapture(self.src)
            
            if self.cap and self.cap.isOpened():
                ret, test_frame = self.cap.read()
                if ret:
                    print(f"[SUCCESS] Camera '{self.name}' opened on attempt {attempt + 1}")
                    break
                else:
                    self.cap.release()
                    self.cap = None
            
            if attempt < max_retries - 1:
                print(f"[RETRY] Attempt {attempt + 1} failed, retrying...")
                time.sleep(1)

        if not self.cap or not self.cap.isOpened():
            print(f"[ERROR] Failed to open camera: {self.src}")
            self.update_camera_status("Failed to Open", "gray", is_live=False)
            
            error_frame = 100 * np.ones((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, "Camera Error", (180, 220), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
            cv2.putText(error_frame, self.name, (200, 260), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
            
            with shared_frames[self.camera_id]["lock"]:
                shared_frames[self.camera_id]["frame"] = error_frame
            
            with camera_lock:
                if self.camera_id in CAMERA_DEFINITIONS:
                    CAMERA_DEFINITIONS[self.camera_id]['isLive'] = False
                    CAMERA_DEFINITIONS[self.camera_id]['thread_instance'] = None
            return

        is_video_file = isinstance(self.src, str) and not str(self.src).isdigit() and os.path.exists(self.src)
        
        print(f"[INIT] {self.name}: Source type check: src={self.src}, is_str={isinstance(self.src, str)}, is_digit={str(self.src).isdigit()}, exists={os.path.exists(self.src) if isinstance(self.src, str) else 'N/A'}, is_video_file={is_video_file}")

        if not is_video_file:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            print(f"[INIT] {self.name}: Configured as live camera (buffersize=1, fps=30)")
        else:
            video_fps = self.cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"[INIT] {self.name}: Configured as video file (FPS={video_fps}, Frames={total_frames})")

        ret, first_frame = self.cap.read()
        if ret:
            first_frame = cv2.resize(first_frame, (640, 480))
            with shared_frames[self.camera_id]["lock"]:
                shared_frames[self.camera_id]["frame"] = first_frame
            print(f"[INIT] {self.name}: Read first frame successfully")
            
            if is_video_file:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                print(f"[INIT] {self.name}: Reset video to frame 0 for processing loop")
        else:
            print(f"[INIT] {self.name}: ERROR - Could not read first frame!")
        
        self.is_running = True
        
        # Immediately update status to ensure it's visible
        with camera_lock:
            CAMERA_DEFINITIONS[self.camera_id]['isLive'] = True
            CAMERA_DEFINITIONS[self.camera_id]['thread_instance'] = self
        
        self.update_camera_status("Active", "green", is_live=True, person_count=0)
        
        print(f"[SUCCESS] Camera '{self.name}' started successfully with ID={self.camera_id}")

        consecutive_failures = 0
        max_failures = 30
        
        # Performance options
        frame_skip = 0
        self.pose_process_interval = GLOBAL_SETTINGS.get('pose_process_interval', 1)
        self.force_detection_next_frame = True
        frame_skip_counter = 0
        
        print(f"[START] {self.name}: Starting main loop with pose_process_interval={self.pose_process_interval}, force_detection_next_frame=True")
        
        try:
            while self.is_running:
                start_time = time.time()
                
                # Skip frames for video files to improve performance
                if is_video_file and frame_skip > 0:
                    for _ in range(frame_skip):
                        self.cap.grab()
                    frame_skip_counter += 1
                
                ret, frame = self.cap.read()
                
                # Fix for video files freezing after fall detection
                if is_video_file and not ret:
                    # Reset video to beginning when it ends
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    # Reset fall detection state when video loops
                    self.consecutive_fall_frames = 0
                    self.alert_sent_for_current_fall = False
                    self.current_fall_id = None
                    
                    # Clear ALL people trackers and person IDs on video loop
                    self.people_trackers.clear()
                    if hasattr(self, 'person_pose_sequences'):
                        self.person_pose_sequences.clear()
                    if hasattr(self, 'person_fall_states'):
                        self.person_fall_states.clear()
                    self.next_person_id = 1
                    
                    # FORCE detection on next frame
                    self.force_detection_next_frame = True
                    print(f"[VIDEO-LOOP] {self.name}: Video looped - cleared all trackers, reset person IDs, forcing detection")

                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        print(f"[ERROR] {self.name}: Too many consecutive failures")
                        break
                    time.sleep(0.05)
                    continue
                
                consecutive_failures = 0
                
                frame = cv2.resize(frame, (640, 480))
                
                # Add frame to buffer for pre-fall recording
                FRAME_BUFFERS[self.camera_id].append(frame.copy())
                
                # MULTI-PERSON DETECTION: YOLOv11 detects all people per frame
                # Optimize: detect every 2 frames for better FPS
                people = []
                if self.force_detection_next_frame or (self.frame_count % 2 == 0):
                    # Use YOLOv11-Pose for multi-person detection
                    people = detect_multiple_people(frame, None, use_hog=False)
                    # Clear the force flag after detection
                    self.force_detection_next_frame = False
                
                # Track and process multiple people
                current_time = time.time()
                tracked_people_ids = set()
                detected_fall = False
                primary_person_fall_prob = 0.0
                
                # Reset person ID counter when no people are detected or after many frames
                if len(people) == 0 and len(self.people_trackers) == 0:
                    self.next_person_id = 1
                elif self.next_person_id > 100:
                    # Clean up all old trackers and reset
                    self.people_trackers.clear()
                    self.person_pose_sequences.clear() if hasattr(self, 'person_pose_sequences') else None
                    self.person_fall_states.clear() if hasattr(self, 'person_fall_states') else None
                    self.next_person_id = 1
                
                # If we have new detections, match them to existing people
                if len(people) > 0:
                    for person_idx, person in enumerate(people):
                        person_bbox = person['bbox']
                        person_keypoints = person['keypoints']  # YOLOv11 keypoints
                        
                        # Match person to existing tracker or create new one
                        person_id = self._match_person(person_bbox)
                        
                        if person_id is None:
                            # New person detected
                            person_id = self.next_person_id
                            self.next_person_id += 1
                        
                        tracked_people_ids.add(person_id)
                        
                        # Smooth the bounding box to reduce flickering
                        new_bbox = person_bbox
                        if person_id in self.people_trackers and 'bbox' in self.people_trackers[person_id]:
                            old_bbox = self.people_trackers[person_id]['bbox']
                            new_bbox = (
                                int(old_bbox[0] * 0.65 + person_bbox[0] * 0.35),
                                int(old_bbox[1] * 0.65 + person_bbox[1] * 0.35),
                                int(old_bbox[2] * 0.65 + person_bbox[2] * 0.35),
                                int(old_bbox[3] * 0.65 + person_bbox[3] * 0.35)
                            )
                        
                        # Update tracker state - Store keypoints for skeleton drawing
                        self.people_trackers[person_id] = {
                            'center': (person['x'], person['y']),
                            'bbox': new_bbox,
                            'last_seen': current_time,
                            'keypoints': person_keypoints  # Store YOLOv11 keypoints
                        }
                        
                        # Extract features for this person using YOLOv11 keypoints
                        try:
                            features_8 = extract_8_kinematic_features(person_keypoints, frame.shape[1], frame.shape[0])
                            feature_vec = np.zeros(FEATURE_SIZE, dtype=np.float32)
                            feature_vec[:8] = features_8
                            
                            # Use person's own pose sequence
                            if not hasattr(self, 'person_pose_sequences'):
                                self.person_pose_sequences = {}
                            if person_id not in self.person_pose_sequences:
                                self.person_pose_sequences[person_id] = deque(
                                    [np.zeros(FEATURE_SIZE, dtype=np.float32) for _ in range(self.sequence_length)],
                                    maxlen=self.sequence_length
                                )
                            
                            self.person_pose_sequences[person_id].append(feature_vec)
                            
                        except Exception as e:
                            feature_vec = np.zeros(FEATURE_SIZE, dtype=np.float32)
                        
                        # Predict fall for this person
                        is_falling_person, fall_prob_person = self._predict_fall_for_person(person_id, feature_vec)
                        
                        if person_idx == 0:  # Store primary person's data for visualization
                            primary_person_fall_prob = fall_prob_person
                            self.latest_fall_prob = fall_prob_person
                            self.latest_features = feature_vec
                            self.latest_keypoints = person_keypoints
                        
                        # Handle fall detection per person - REQUIRE 7+ FRAMES for confirmation
                        if is_falling_person:
                            if not hasattr(self, 'person_fall_states'):
                                self.person_fall_states = {}
                            if person_id not in self.person_fall_states:
                                self.person_fall_states[person_id] = {'frames': 0, 'alerted': False}
                            
                            self.person_fall_states[person_id]['frames'] += 1
                            
                            # Confirm fall after 7 consecutive frames
                            if self.person_fall_states[person_id]['frames'] >= 7:
                                detected_fall = True
                                
                                # Send alert for this person
                                self.update_camera_status("FALL DETECTED", "red", last_alert=current_time, is_live=True, person_count=len(people))
                                self.trigger_website_alert(fall_prob_person, person_id)
                                self.send_fall_alert_with_video(frame.copy(), fall_prob_person, person_id)
                                
                                if not self.person_fall_states[person_id]['alerted']:
                                    self.create_incident_report(frame.copy(), fall_prob_person, person_id)
                                    self.person_fall_states[person_id]['alerted'] = True
                        else:
                            # Person not falling - reset fall state
                            if hasattr(self, 'person_fall_states') and person_id in self.person_fall_states:
                                self.person_fall_states[person_id]['frames'] = 0
                                self.person_fall_states[person_id]['alerted'] = False
                else:
                    # No new detection on this frame - keep existing tracked people
                    for person_id in list(self.people_trackers.keys()):
                        tracked_people_ids.add(person_id)
                        # Update last_seen to keep tracker alive
                        self.people_trackers[person_id]['last_seen'] = current_time
                
                # Remove people not seen for timeout
                stale_people = []
                for person_id, tracker in self.people_trackers.items():
                    if current_time - tracker['last_seen'] > self.person_timeout:
                        stale_people.append(person_id)
                
                for person_id in stale_people:
                    del self.people_trackers[person_id]
                    if person_id in self.person_pose_sequences:
                        del self.person_pose_sequences[person_id]
                    if person_id in self.person_fall_states:
                        del self.person_fall_states[person_id]
                
                # Reset person ID counter if all people are gone
                if len(self.people_trackers) == 0 and len(people) == 0:
                    self.next_person_id = 1
                
                # Update status
                if not detected_fall:
                    # Use tracked_people_ids count
                    num_people_tracked = len(tracked_people_ids)
                    if num_people_tracked > 0:
                        status_msg = f"{num_people_tracked} Person(s) - Normal" if num_people_tracked > 1 else "Normal"
                        self.update_camera_status(status_msg, "green", is_live=True, person_count=num_people_tracked)
                    else:
                        self.update_camera_status("No People Detected", "gray", is_live=True, person_count=0)
                
                # Apply privacy mode to frame before skeleton drawing for certain modes
                privacy_mode = GLOBAL_SETTINGS.get('privacy_mode', 'full_video')
                
                # Draw frame with skeleton overlay only (no bounding boxes)
                # For skeleton_only and alerts_only, start with black frame so skeleton shows clearly
                if privacy_mode == "skeleton_only":
                    processed = np.zeros_like(frame)  # Black frame for skeleton only
                elif privacy_mode == "alerts_only":
                    processed = np.zeros_like(frame)  # Black frame for alerts only (no skeleton drawn)
                elif privacy_mode == "blurred":
                    processed = cv2.GaussianBlur(frame, (51, 51), 30)  # Blur before skeleton
                else:  # full_video
                    processed = frame.copy()
                
                # Color palette for different people
                person_colors = [
                    (0, 255, 0),      # Green
                    (255, 0, 0),      # Blue
                    (0, 255, 255),    # Yellow
                    (255, 0, 255),    # Magenta
                    (255, 255, 0),    # Cyan
                    (128, 0, 255),    # Purple
                    (255, 128, 0),    # Orange
                    (0, 128, 255),    # Red-Orange
                ]
                
                # Draw skeleton for ALL detected people (only if not in alerts_only mode)
                if privacy_mode != "alerts_only":
                    for person_idx, person in enumerate(people):
                        person_keypoints = person.get('keypoints')
                        if person_keypoints is not None:
                            # Find person_id if tracked
                            person_id = None
                            person_bbox = person['bbox']
                            for pid, tracker in self.people_trackers.items():
                                tracker_bbox = tracker.get('bbox', (0, 0, 0, 0))
                                # Simple IoU check
                                x1, y1, w1, h1 = person_bbox
                                x2, y2, w2, h2 = tracker_bbox
                                xi1, yi1 = max(x1, x2), max(y1, y2)
                                xi2, yi2 = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
                                if xi2 > xi1 and yi2 > yi1:
                                    intersection = (xi2 - xi1) * (yi2 - yi1)
                                    area1, area2 = w1*h1, w2*h2
                                    union = area1 + area2 - intersection
                                    if union > 0 and intersection / union > 0.5:
                                        person_id = pid
                                        break
                            
                            person_falling = False
                            if person_id and hasattr(self, 'person_fall_states') and person_id in self.person_fall_states:
                                person_falling = self.person_fall_states[person_id]['frames'] >= 7
                            
                            # Get color for this person
                            color_idx = person_idx % len(person_colors)
                            person_color = person_colors[color_idx]
                            
                            # Choose color based on fall status
                            if person_falling:
                                color = (0, 0, 255)  # Red for falling
                            else:
                                color = person_color  # Use assigned color
                            
                            # Draw skeleton directly from YOLOv11 keypoints
                            draw_skeleton_yolo(processed, person_keypoints, color=color, thickness=2)

                with shared_frames[self.camera_id]["lock"]:
                    shared_frames[self.camera_id]["frame"] = processed

                self.frame_count += 1
                if time.time() - self.last_fps_update >= 1.0:
                    self.current_fps = self.frame_count / (time.time() - self.last_fps_update)
                    self.frame_count = 0
                    self.last_fps_update = time.time()
                    # Only print FPS if very low
                    if self.current_fps < 10:
                        print(f"[FPS] {self.name}: {self.current_fps:.1f} fps")
                    
                    # Adaptive frame skipping for video files based on performance
                    if is_video_file:
                        if self.current_fps < 12:
                            # Low FPS - skip MORE frames
                            frame_skip = min(3, frame_skip + 1)
                        elif self.current_fps > 30:
                            # High FPS - skip FEWER frames
                            frame_skip = max(0, frame_skip - 1)

                processing_time = time.time() - start_time
                if is_video_file:
                    fps = self.cap.get(cv2.CAP_PROP_FPS) or INTERNAL_FPS 
                    target_delay = 1.0 / fps
                    sleep_time = max(0, target_delay - processing_time)
                    time.sleep(sleep_time)
                else:
                    target_delay = 1.0 / INTERNAL_FPS
                    sleep_time = max(0, target_delay - processing_time)
                    time.sleep(sleep_time)

        except Exception as e:
            print(f"[ERROR] Camera processor crashed: {e}")
            print(f"[ERROR] Camera: {self.name} (ID: {self.camera_id})")
            import traceback
            traceback.print_exc()
        finally:
            try:
                if self.cap: 
                    self.cap.release()
            except Exception as e:
                print(f"[ERROR] Cleanup error: {e}")
            
            with camera_lock:
                if self.camera_id in CAMERA_STATUS: 
                    del CAMERA_STATUS[self.camera_id]
                if self.camera_id in CAMERA_DEFINITIONS: 
                    CAMERA_DEFINITIONS[self.camera_id]['isLive'] = False
                    CAMERA_DEFINITIONS[self.camera_id]['thread_instance'] = None
            
            # Clean up frame buffer
            if self.camera_id in FRAME_BUFFERS:
                del FRAME_BUFFERS[self.camera_id]
                
            print(f"[INFO] Camera '{self.name}' (ID: {self.camera_id}) stopped")

    def create_incident_report(self, frame, fall_probability, person_id=None):
        """Create an incident report for the detected fall - SUPPORTS MULTIPLE PEOPLE"""
        incident_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Save snapshot as file instead of base64 to reduce JSON size
        snapshot_filename = f"{incident_id}_{timestamp.replace(':', '-').replace(' ', '_')}.jpg"
        snapshot_path = os.path.join(SNAPSHOTS_DIR, snapshot_filename)
        cv2.imwrite(snapshot_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        
        person_label = f"Person #{person_id}" if person_id else "Detected Person"
        
        incident = {
            "id": incident_id,
            "fall_id": f"{self.camera_id}_person{person_id}_{int(time.time())}" if person_id else f"{self.camera_id}_{int(time.time())}",
            "timestamp": timestamp,
            "camera_name": self.name,
            "camera_id": self.camera_id,
            "person_id": person_id,
            "person_label": person_label,
            "confidence": float(fall_probability),
            "severity": "HIGH" if fall_probability > 0.8 else "MEDIUM" if fall_probability > 0.6 else "LOW",
            "snapshot_file": snapshot_filename,
            "notes": "",
            "location": f"{self.name} - {person_label}" if person_id else self.name
        }
        
        INCIDENT_REPORTS.append(incident)
        save_incident_reports()
        
        print(f"[INCIDENT] Created report {incident_id} for fall at {timestamp} - {person_label}")

# MJPEG Stream Generator
def generate_mjpeg(camera_id):
    boundary = b'--frame\r\n'
    
    print(f"[MJPEG] Starting stream for camera_id='{camera_id}'")
    print(f"[MJPEG] Looking for camera_id in shared_frames...")
    
    wait_time = 0
    max_wait = 5
    
    while camera_id not in shared_frames and wait_time < max_wait:
        print(f"[MJPEG] Waiting... ({wait_time}s/{max_wait}s) Cameras available: {list(shared_frames.keys())}")
        time.sleep(0.1)
        wait_time += 0.1
    
    if camera_id not in shared_frames:
        print(f"[MJPEG] ERROR: Camera {camera_id} not found in shared_frames after {max_wait}s!")
        print(f"[MJPEG] Available cameras: {list(shared_frames.keys())}")
        placeholder = 100 * np.ones((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Camera Not Available", (150, 220), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(placeholder, f"ID: {camera_id}", (200, 260), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
        
        ret, jpeg = cv2.imencode('.jpg', placeholder)
        frame_bytes = jpeg.tobytes()
        yield boundary + b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        return
    
    print(f"[MJPEG] Found camera {camera_id}, starting stream...")
    frame_count = 0
    while camera_id in shared_frames:
        frame_count += 1
        frame_data = shared_frames[camera_id]
        with frame_data["lock"]:
            frame = frame_data["frame"].copy() if frame_data["frame"] is not None else None
        
        if frame is None:
            placeholder = 100 * np.ones((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Initializing...", (180, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            ret, jpeg = cv2.imencode('.jpg', placeholder)
            frame_bytes = jpeg.tobytes()
        else:
            # Optimized: Lower quality for faster streaming
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
            ret, jpeg = cv2.imencode('.jpg', frame, encode_param)
            frame_bytes = jpeg.tobytes()

        yield boundary + b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        if frame_count % 30 == 0:
            print(f"[MJPEG] Streaming {camera_id}: frame {frame_count}")
        time.sleep(0.033)  # ~30 FPS

# Flask Routes
@app.route('/')
def index():
    return send_from_directory('app', 'index.html')

@app.route('/static/<path:filepath>')
def serve_static(filepath):
    """Serve static files (images, CSS, JS)"""
    return send_from_directory('app/static', filepath)

@app.route('/video_feed/<camera_id>')
def video_feed(camera_id):
    print(f"[VIDEO_FEED] Request for camera_id='{camera_id}', type={type(camera_id)}")
    print(f"[VIDEO_FEED] Available cameras in shared_frames: {list(shared_frames.keys())}")
    print(f"[VIDEO_FEED] Available cameras in CAMERA_DEFINITIONS: {list(CAMERA_DEFINITIONS.keys())}")
    return Response(generate_mjpeg(camera_id), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/snapshot/<camera_id>')
def snapshot(camera_id):
    """Return a single JPEG snapshot for a camera to avoid long-lived MJPEG connections from thumbnails."""
    # Wait briefly for shared frame to be initialized
    wait_time = 0.0
    max_wait = 2.0
    while camera_id not in shared_frames and wait_time < max_wait:
        time.sleep(0.05)
        wait_time += 0.05

    if camera_id not in shared_frames:
        print(f"[SNAPSHOT] Camera {camera_id} not in shared_frames after {max_wait}s wait")
        placeholder = 100 * np.ones((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "No Camera", (200, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        ret, jpeg = cv2.imencode('.jpg', placeholder)
        return Response(jpeg.tobytes(), mimetype='image/jpeg')

    frame_data = shared_frames[camera_id]
    with frame_data['lock']:
        frame = None if frame_data['frame'] is None else frame_data['frame'].copy()

    if frame is None:
        print(f"[SNAPSHOT] Camera {camera_id} has no frame yet")
        placeholder = 100 * np.ones((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Initializing...", (160, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        ret, jpeg = cv2.imencode('.jpg', placeholder)
        return Response(jpeg.tobytes(), mimetype='image/jpeg')

    # Optimized: Lower quality for faster thumbnail loading
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 65]
    ret, jpeg = cv2.imencode('.jpg', frame, encode_param)
    return Response(jpeg.tobytes(), mimetype='image/jpeg')

# Add website alerts API endpoint
@app.route('/api/alerts/active', methods=['GET'])
def api_get_active_alerts():
    """
    Get active website alerts for all cameras.
    Supports multiple people per camera with individual tracking IDs.
    Returns alerts that are less than 35 seconds old.
    """
    current_time = time.time()
    active_alerts = []
    
    with website_alerts_lock:
        # Remove old alerts (older than 35 seconds)
        expired_keys = []
        for alert_key, alert_data in WEBSITE_ALERTS.items():
            # Keep alerts that are recent
            if current_time - alert_data['timestamp'] < 35:
                # Build complete alert info
                alert_info = {
                    'camera_id': alert_data['camera_id'],
                    'camera_name': alert_data['camera_name'],
                    'confidence': alert_data['confidence'],
                    'timestamp': alert_data['timestamp'],
                    'alert_id': alert_data.get('alert_id', alert_key),
                    'person_id': alert_data.get('person_id'),
                    'fall_severity': alert_data.get('fall_severity', 'MEDIUM')
                }
                active_alerts.append(alert_info)
            else:
                # Mark expired alert for cleanup
                expired_keys.append(alert_key)
        
        # Clean up expired alerts
        for key in expired_keys:
            del WEBSITE_ALERTS[key]
    
    return jsonify({
        "success": True,
        "alerts": active_alerts,
        "count": len(active_alerts),
        "timestamp": current_time
    })

@app.route('/api/admin/login', methods=['POST'])
def api_admin_login():
    data = request.get_json() or {}
    password = data.get('password', '')
    
    if password == ADMIN_PASSWORD:
        session['admin_authenticated'] = True
        return jsonify({"success": True, "message": "Login successful"})
    else:
        return jsonify({"success": False, "message": "Invalid password"}), 401

@app.route('/api/admin/logout', methods=['POST'])
def api_admin_logout():
    session.pop('admin_authenticated', None)
    return jsonify({"success": True, "message": "Logged out"})

@app.route('/api/admin/check', methods=['GET'])
def api_admin_check():
    is_authenticated = session.get('admin_authenticated', False)
    return jsonify({"authenticated": is_authenticated})

@app.route('/api/settings', methods=['GET', 'POST'])
def api_settings():
    # GET requests are allowed without authentication (for loading current settings)
    # POST requests still require authentication for most settings, but privacy settings are allowed
    
    if request.method == 'POST':
        data = request.get_json() or {}
        message = []
        
        # Check if user is trying to modify sensitive settings (requires auth)
        has_sensitive_changes = any(key in data for key in ['fall_threshold', 'fall_delay_seconds', 'alert_cooldown_seconds'])
        
        if has_sensitive_changes and not session.get('admin_authenticated', False):
            return jsonify({"success": False, "message": "Authentication required for these settings"}), 401
        
        new_threshold = data.get('fall_threshold')
        if new_threshold is not None:
            try:
                new_threshold = float(new_threshold)
                if 0.0 <= new_threshold <= 1.0:
                    GLOBAL_SETTINGS['fall_threshold'] = new_threshold
                    message.append("Threshold updated")
                else:
                    return jsonify({"success": False, "message": "Threshold must be 0.0-1.0"}), 400
            except ValueError:
                return jsonify({"success": False, "message": "Invalid threshold value"}), 400
                
        new_delay = data.get('fall_delay_seconds')
        if new_delay is not None:
            try:
                new_delay = int(new_delay)
                if 1 <= new_delay <= 10: 
                    GLOBAL_SETTINGS['fall_delay_seconds'] = new_delay
                    message.append("Delay updated")
                    
                    with camera_lock:
                        for cam_def in CAMERA_DEFINITIONS.values():
                            processor = cam_def.get('thread_instance')
                            if processor and processor.is_running:
                                processor.update_fall_timer_threshold()
                else:
                    return jsonify({"success": False, "message": "Delay must be 1-10 seconds"}), 400
            except ValueError:
                return jsonify({"success": False, "message": "Invalid delay value"}), 400
        
        new_cooldown = data.get('alert_cooldown_seconds')
        if new_cooldown is not None:
            try:
                new_cooldown = int(new_cooldown)
                if 0 <= new_cooldown <= 300:
                    GLOBAL_SETTINGS['alert_cooldown_seconds'] = new_cooldown
                    message.append("Cooldown updated")
                else:
                    return jsonify({"success": False, "message": "Cooldown must be 0-300 seconds"}), 400
            except ValueError:
                return jsonify({"success": False, "message": "Invalid cooldown value"}), 400
        
        new_privacy_mode = data.get('privacy_mode')
        if new_privacy_mode in ['full_video', 'skeleton_only', 'blurred', 'alerts_only']:
            GLOBAL_SETTINGS['privacy_mode'] = new_privacy_mode
            message.append("Privacy mode updated")
        
        new_buffer_seconds = data.get('pre_fall_buffer_seconds')
        if new_buffer_seconds is not None:
            try:
                new_buffer_seconds = int(new_buffer_seconds)
                if 1 <= new_buffer_seconds <= 30:
                    GLOBAL_SETTINGS['pre_fall_buffer_seconds'] = new_buffer_seconds
                    global BUFFER_SIZE
                    BUFFER_SIZE = new_buffer_seconds * INTERNAL_FPS
                    message.append("Pre-fall buffer updated")
                else:
                    return jsonify({"success": False, "message": "Buffer must be 1-30 seconds"}), 400
            except ValueError:
                return jsonify({"success": False, "message": "Invalid buffer value"}), 400

        return jsonify({"success": True, "message": " ".join(message) if message else "No changes", "settings": GLOBAL_SETTINGS})
    
    response_data = {"success": True, "settings": GLOBAL_SETTINGS}
    
    # Add Telegram info if configured
    if TELEGRAM_BOT_TOKEN:
        response_data['telegram_token'] = True
        bot_info = get_bot_info()
        if bot_info:
            response_data['bot_username'] = bot_info.get('username', '')
            response_data['telegram_bot_name'] = bot_info.get('first_name', 'Bot')
    
    return jsonify(response_data)

# Telegram API Routes
@app.route('/api/telegram/set_token', methods=['POST'])
def api_telegram_set_token():
    if not session.get('admin_authenticated', False):
        return jsonify({"success": False, "message": "Unauthorized"}), 401
    
    global TELEGRAM_BOT_TOKEN
    
    data = request.get_json() or {}
    token = data.get('token', '').strip()
    
    if not token:
        return jsonify({"success": False, "message": "Token is required"}), 400
    
    # Verify token by getting bot info
    test_url = f"https://api.telegram.org/bot{token}/getMe"
    try:
        response = requests.get(test_url, timeout=10)
        if response.status_code == 200:
            bot_data = response.json().get('result', {})
            TELEGRAM_BOT_TOKEN = token
            return jsonify({
                "success": True,
                "message": "Telegram bot token saved",
                "bot_username": bot_data.get('username', '')
            })
        else:
            return jsonify({"success": False, "message": "Invalid bot token"}), 400
    except Exception as e:
        return jsonify({"success": False, "message": f"Failed to verify token: {str(e)}"}), 400

@app.route('/api/telegram/subscribers', methods=['GET'])
def api_telegram_subscribers():
    # Return current subscribers (already loaded from file on startup)
    with telegram_lock:
        return jsonify({"success": True, "subscribers": TELEGRAM_SUBSCRIBERS.copy()})

@app.route('/api/telegram/add_subscriber', methods=['POST'])
def api_telegram_add_subscriber():
    data = request.get_json() or {}
    chat_id = data.get('chat_id', '').strip()
    name = data.get('name', 'Manual Entry').strip()
    
    if not chat_id:
        return jsonify({"success": False, "message": "Chat ID is required"}), 400
    
    with telegram_lock:
        # Check if already exists
        if any(sub['chat_id'] == chat_id for sub in TELEGRAM_SUBSCRIBERS):
            return jsonify({"success": False, "message": "Subscriber already exists"}), 400
        
        TELEGRAM_SUBSCRIBERS.append({
            'chat_id': chat_id,
            'name': name,
            'username': ''
        })
    
    # Save to persistent storage
    save_subscribers_to_file()
    
    # Send welcome message
    if TELEGRAM_BOT_TOKEN:
        send_telegram_message(
            chat_id,
            f"✅ <b>Added to FallGuard</b>\n\nYou will now receive fall detection alerts."
        )
    
    return jsonify({"success": True, "message": "Subscriber added"})

@app.route('/api/telegram/remove_subscriber', methods=['POST'])
def api_telegram_remove_subscriber():
    data = request.get_json() or {}
    chat_id = data.get('chat_id', '').strip()
    
    if not chat_id:
        return jsonify({"success": False, "message": "Chat ID is required"}), 400
    
    with telegram_lock:
        TELEGRAM_SUBSCRIBERS[:] = [sub for sub in TELEGRAM_SUBSCRIBERS if sub['chat_id'] != chat_id]
        # Add to blocklist to prevent re-adding if they send /start again
        if chat_id not in TELEGRAM_BLOCKED_LIST:
            TELEGRAM_BLOCKED_LIST.append(chat_id)
    
    # Save to persistent storage
    save_subscribers_to_file()
    save_blocked_list_to_file()
    
    return jsonify({"success": True, "message": "Subscriber removed"})

@app.route('/api/telegram/test_alert', methods=['POST'])
def api_telegram_test_alert():
    if not session.get('admin_authenticated', False):
        return jsonify({"success": False, "message": "Unauthorized"}), 401
    
    if not TELEGRAM_BOT_TOKEN:
        return jsonify({"success": False, "message": "Telegram bot not configured"}), 400
    
    if not TELEGRAM_SUBSCRIBERS:
        return jsonify({"success": False, "message": "No subscribers"}), 400
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = (
        f"🧪 <b>TEST ALERT</b> 🧪\n\n"
        f"This is a test notification from FallGuard.\n\n"
        f"🕐 <b>Time:</b> {timestamp}\n"
        f"✅ Your notifications are working correctly!"
    )
    
    sent_count = 0
    with telegram_lock:
        for subscriber in TELEGRAM_SUBSCRIBERS:
            try:
                if send_telegram_message(subscriber['chat_id'], message):
                    sent_count += 1
            except Exception as e:
                print(f"[ERROR] Failed to send test to {subscriber['name']}: {e}")
    
    return jsonify({
        "success": True,
        "message": "Test alerts sent",
        "sent_count": sent_count
    })

@app.route('/api/telegram/blocked', methods=['GET'])
def api_telegram_blocked():
    with telegram_lock:
        return jsonify({"success": True, "blocked": TELEGRAM_BLOCKED_LIST.copy()})

@app.route('/api/telegram/unblock', methods=['POST'])
def api_telegram_unblock():
    data = request.get_json() or {}
    chat_id = data.get('chat_id', '').strip()
    
    if not chat_id:
        return jsonify({"success": False, "message": "Chat ID is required"}), 400
    
    with telegram_lock:
        if chat_id in TELEGRAM_BLOCKED_LIST:
            TELEGRAM_BLOCKED_LIST.remove(chat_id)
            save_blocked_list_to_file()
            return jsonify({"success": True, "message": "User unblocked"})
        else:
            return jsonify({"success": False, "message": "User not in blocked list"}), 400

# Camera Management Routes
@app.route('/api/cameras', methods=['GET'])
def api_get_cameras():
    cameras = []
    with camera_lock:
        for cam_id, cam_def in CAMERA_DEFINITIONS.items():
            processor = cam_def.get('thread_instance')
            
            # Check if processor is actually running
            is_actually_live = False
            if processor is not None:
                try:
                    is_actually_live = processor.is_running and processor.is_alive()
                except:
                    is_actually_live = False
            
            # Get status, with fallback
            status = CAMERA_STATUS.get(cam_id, {})
            
            # If no status, create default
            if not status:
                status = {
                    "status": "Offline",
                    "color": "gray",
                    "isLive": False,
                    "confidence_score": 0.0,
                    "fps": 0.0
                }
            
            # Use actual live state
            actual_is_live = is_actually_live and cam_id in shared_frames
            
            cameras.append({
                "id": cam_id,
                "name": cam_def['name'],
                "source": str(cam_def['source']),
                "isLive": actual_is_live,
                "status": status.get('status', 'Offline') if actual_is_live else "Offline",
                "color": status.get('color', 'gray') if actual_is_live else "gray",
                "lastAlert": status.get('lastAlert', 'N/A'),
                "confidence_score": float(status.get('confidence_score', 0.0)),
                "model_threshold": GLOBAL_SETTINGS['fall_threshold'],
                "fps": float(status.get('fps', 0.0))
            })
    
    return jsonify({"success": True, "cameras": cameras})

@app.route('/api/cameras/all_definitions', methods=['GET'])
def api_get_all_definitions():
    definitions = []
    with camera_lock:
        for cam_id, cam_def in CAMERA_DEFINITIONS.items():
            processor = cam_def.get('thread_instance')
            is_live = False
            if processor is not None:
                try:
                    is_live = processor.is_running and processor.is_alive()
                except:
                    is_live = False
            
            status = CAMERA_STATUS.get(cam_id, {
                "confidence_score": 0.0,
                "fps": 0,
                "status": "Offline"
            })
            
            definitions.append({
                "id": cam_id,
                "name": cam_def['name'],
                "source": str(cam_def['source']),
                "isLive": is_live,
                "confidence_score": status.get('confidence_score', 0.0),
                "fps": status.get('fps', 0),
                "status": status.get('status', 'Offline')
            })
    return jsonify({"success": True, "definitions": definitions})

@app.route('/api/cameras/add', methods=['POST'])
def api_add_camera():
    data = request.get_json()
    name = data.get('name')
    source_str = data.get('source')
    
    if not name or source_str is None:
        return jsonify({"success": False, "message": "Name and source required"}), 400

    try:
        source = int(source_str)
    except ValueError:
        source = source_str
    
    camera_id = f"cam_{str(uuid.uuid4())[:8]}"
    
    processor = CameraProcessor(camera_id=camera_id, src=source, name=name, device=device)
    processor.start()
    
    with camera_lock:
        CAMERA_DEFINITIONS[camera_id] = {
            "name": name, 
            "source": source, 
            "isLive": True,
            "thread_instance": processor
        }

    return jsonify({"success": True, "message": f"Camera '{name}' added", "camera_id": camera_id})

@app.route('/api/cameras/stop/<camera_id>', methods=['POST'])
def api_stop_camera(camera_id):
    with camera_lock:
        if camera_id not in CAMERA_DEFINITIONS:
            return jsonify({"success": False, "message": "Camera not found"}), 404

        cam_def = CAMERA_DEFINITIONS[camera_id]
        processor = cam_def.get('thread_instance')

        if processor and processor.is_running:
            processor.is_running = False
            processor.join(timeout=3)

        CAMERA_DEFINITIONS[camera_id]['isLive'] = False
        CAMERA_DEFINITIONS[camera_id]['thread_instance'] = None
        
        if camera_id in CAMERA_STATUS:
            del CAMERA_STATUS[camera_id]

    return jsonify({"success": True, "message": "Camera stopped"})

@app.route('/api/cameras/remove/<camera_id>', methods=['DELETE'])
def api_remove_camera(camera_id):
    with camera_lock:
        if camera_id not in CAMERA_DEFINITIONS:
            return jsonify({"success": False, "message": "Camera not found"}), 404

        cam_def = CAMERA_DEFINITIONS[camera_id]
        processor = cam_def.get('thread_instance')
        if processor and processor.is_running:
            processor.is_running = False
            processor.join(timeout=3)

        source = cam_def['source']
        if isinstance(source, str) and source.startswith(UPLOAD_FOLDER):
            try:
                if os.path.exists(source):
                    os.remove(source)
                    print(f"[INFO] Deleted video file: {source}")
            except Exception as e:
                print(f"[WARNING] Could not delete file {source}: {e}")

        del CAMERA_DEFINITIONS[camera_id]
        if camera_id in CAMERA_STATUS:
            del CAMERA_STATUS[camera_id]
        if camera_id in shared_frames:
            del shared_frames[camera_id]

    return jsonify({"success": True, "message": "Camera removed"})

@app.route('/api/cameras/add_existing', methods=['POST'])
def api_add_existing_camera():
    data = request.get_json()
    camera_id = data.get('camera_id')
    
    if not camera_id:
        return jsonify({"success": False, "message": "Camera ID required"}), 400

    with camera_lock:
        if camera_id not in CAMERA_DEFINITIONS:
            return jsonify({"success": False, "message": "Camera not found"}), 404
        
        cam_def = CAMERA_DEFINITIONS[camera_id]
        
        if cam_def.get('thread_instance') and cam_def['thread_instance'].is_running:
            return jsonify({"success": False, "message": "Camera already running"}), 400
        
        src_type = cam_def['source']
        try:
            if isinstance(src_type, str) and src_type.isdigit():
                src_type = int(src_type)
        except:
            pass

        processor = CameraProcessor(camera_id=camera_id, src=src_type, name=cam_def['name'], device=device)
        processor.start()
        
        CAMERA_DEFINITIONS[camera_id]['isLive'] = True
        CAMERA_DEFINITIONS[camera_id]['thread_instance'] = processor
        
    return jsonify({"success": True, "message": "Camera restarted"})

@app.route('/api/cameras/upload', methods=['POST'])
def api_upload_video():
    if 'video_file' not in request.files:
        return jsonify({"success": False, "message": "No video file provided"}), 400

    video_file = request.files['video_file']
    name = request.form.get('name', 'Uploaded Video')

    if video_file.filename == '':
        return jsonify({"success": False, "message": "No selected file"}), 400

    # Save the uploaded file with absolute path
    filename = secure_filename(video_file.filename)
    # Use absolute path to ensure the processor can find it
    file_path = os.path.abspath(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    video_file.save(file_path)
    print(f"[UPLOAD] Saved video file: {file_path}")

    # Add the uploaded video as a new camera source
    camera_id = f"cam_{str(uuid.uuid4())[:8]}"
    print(f"[UPLOAD] Creating new camera: {camera_id}, name='{name}', source='{file_path}'")
    
    with camera_lock:
        CAMERA_DEFINITIONS[camera_id] = {
            "name": name,
            "source": file_path,  # Use absolute path
            "isLive": False,
            "thread_instance": None
        }
    
    # Start processing the uploaded video
    processor = CameraProcessor(camera_id=camera_id, src=file_path, name=name, device=device)
    processor.start()
    print(f"[UPLOAD] Started processor thread for camera: {camera_id}")

    with camera_lock:
        CAMERA_DEFINITIONS[camera_id] = {
            "name": name,
            "source": file_path,  # Use absolute path
            "isLive": True,
            "thread_instance": processor
        }
    
    print(f"[UPLOAD] Upload complete. Camera {camera_id} is now live")
    return jsonify({"success": True, "message": f"Video '{name}' uploaded successfully", "camera_id": camera_id})

# Incident Reports Routes
@app.route('/api/incidents', methods=['GET'])
def api_get_incidents():
    if not session.get('admin_authenticated', False):
        return jsonify({"success": False, "message": "Unauthorized"}), 401
    
    return jsonify({"success": True, "incidents": INCIDENT_REPORTS})

@app.route('/api/incidents/<incident_id>', methods=['GET'])
def api_get_incident(incident_id):
    if not session.get('admin_authenticated', False):
        return jsonify({"success": False, "message": "Unauthorized"}), 401
    
    incident = next((inc for inc in INCIDENT_REPORTS if inc['id'] == incident_id), None)
    if not incident:
        return jsonify({"success": False, "message": "Incident not found"}), 404
    
    # Ensure required fields for PDF preview
    if 'camera' not in incident:
        incident['camera'] = incident.get('camera_name', 'Unknown')
    if 'location' not in incident:
        incident['location'] = incident.get('camera_name', 'Unknown')
    
    return jsonify({"success": True, "incident": incident})

@app.route('/api/incidents/<incident_id>/pdf', methods=['GET'])
def api_generate_incident_pdf(incident_id):
    if not session.get('admin_authenticated', False):
        return jsonify({"success": False, "message": "Unauthorized"}), 401
    
    incident = next((inc for inc in INCIDENT_REPORTS if inc['id'] == incident_id), None)
    if not incident:
        return jsonify({"success": False, "message": "Incident not found"}), 404
    
    # Create PDF
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # Define colors
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    
    # Header - Professional Layout
    pdf.setFont("Helvetica-Bold", 24)
    pdf.drawString(50, height - 50, "FALL DETECTION INCIDENT REPORT")
    
    # Horizontal line
    pdf.setStrokeColor(colors.HexColor('#1f2937'))
    pdf.setLineWidth(2)
    pdf.line(50, height - 60, width - 50, height - 60)
    
    # System Info Header
    pdf.setFont("Helvetica", 10)
    pdf.setFillColor(colors.HexColor('#6b7280'))
    pdf.drawString(50, height - 75, f"FallGuard AI Fall Detection System • Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Main Content Area
    y_position = height - 110
    
    # Incident Header Box
    pdf.setFillColor(colors.HexColor('#f3f4f6'))
    pdf.rect(50, y_position - 60, width - 100, 50, fill=1, stroke=1)
    
    pdf.setFont("Helvetica-Bold", 12)
    pdf.setFillColor(colors.HexColor('#1f2937'))
    pdf.drawString(60, y_position - 20, f"Incident ID: {incident['id']}")
    pdf.drawString(60, y_position - 35, f"Severity Level: {incident['severity']}")
    pdf.drawString(350, y_position - 20, f"Timestamp: {incident['timestamp']}")
    pdf.drawString(350, y_position - 35, f"Camera: {incident['camera_name']}")
    
    y_position -= 80
    
    # Details Section
    pdf.setFont("Helvetica-Bold", 14)
    pdf.setFillColor(colors.HexColor('#1f2937'))
    pdf.drawString(50, y_position, "INCIDENT DETAILS")
    
    pdf.setLineWidth(1)
    pdf.setStrokeColor(colors.HexColor('#d1d5db'))
    pdf.line(50, y_position - 5, width - 50, y_position - 5)
    
    y_position -= 25
    
    # Details Grid
    pdf.setFont("Helvetica", 11)
    pdf.setFillColor(colors.HexColor('#6b7280'))
    
    details = [
        ("Location:", incident['location']),
        ("Confidence Score:", f"{(incident['confidence'] * 100):.1f}%"),
        ("Severity:", incident['severity']),
        ("Detection Status:", "Fall Detected - High Priority"),
    ]
    
    for label, value in details:
        pdf.setFont("Helvetica-Bold", 11)
        pdf.setFillColor(colors.HexColor('#1f2937'))
        pdf.drawString(60, y_position, label)
        
        pdf.setFont("Helvetica", 11)
        pdf.setFillColor(colors.HexColor('#374151'))
        pdf.drawString(200, y_position, str(value))
        
        y_position -= 20
    
    y_position -= 10
    
    # Add snapshot if available
    if incident.get('snapshot_file'):
        try:
            snapshot_path = os.path.join(SNAPSHOTS_DIR, incident['snapshot_file'])
            if os.path.exists(snapshot_path):
                # Snapshot Section Header
                pdf.setFont("Helvetica-Bold", 14)
                pdf.setFillColor(colors.HexColor('#1f2937'))
                pdf.drawString(50, y_position, "INCIDENT SNAPSHOT")
                
                pdf.setLineWidth(1)
                pdf.setStrokeColor(colors.HexColor('#d1d5db'))
                pdf.line(50, y_position - 5, width - 50, y_position - 5)
                
                y_position -= 20
                
                # Add image with border
                img = ImageReader(snapshot_path)
                img_width = 300
                img_height = 225
                
                # Image border
                pdf.setStrokeColor(colors.HexColor('#d1d5db'))
                pdf.setLineWidth(1)
                pdf.rect(60, y_position - img_height - 10, img_width, img_height, stroke=1)
                
                # Draw image
                pdf.drawImage(img, 62, y_position - img_height - 8, width=img_width - 4, height=img_height - 4)
                
                y_position -= img_height - 10
        except Exception as e:
            pdf.setFont("Helvetica", 10)
            pdf.setFillColor(colors.HexColor('#dc2626'))
            pdf.drawString(60, y_position, f"Error loading snapshot: {str(e)}")
    
    y_position -= 30
    
    # Notes Section (if exists)
    if incident.get('notes'):
        pdf.setFont("Helvetica-Bold", 14)
        pdf.setFillColor(colors.HexColor('#1f2937'))
        pdf.drawString(50, y_position, "ADDITIONAL NOTES")
        
        pdf.setLineWidth(1)
        pdf.setStrokeColor(colors.HexColor('#d1d5db'))
        pdf.line(50, y_position - 5, width - 50, y_position - 5)
        
        y_position -= 20
        
        pdf.setFont("Helvetica", 10)
        pdf.setFillColor(colors.HexColor('#374151'))
        
        # Wrap text if needed
        from textwrap import wrap
        notes_wrapped = wrap(incident['notes'], width=100)
        for line in notes_wrapped:
            pdf.drawString(60, y_position, line)
            y_position -= 15
        
        y_position -= 10
    
    # Notifications Section
    y_position -= 10
    pdf.setFont("Helvetica-Bold", 14)
    pdf.setFillColor(colors.HexColor('#1f2937'))
    pdf.drawString(50, y_position, "TELEGRAM NOTIFICATIONS SENT TO")
    
    pdf.setLineWidth(1)
    pdf.setStrokeColor(colors.HexColor('#d1d5db'))
    pdf.line(50, y_position - 5, width - 50, y_position - 5)
    
    y_position -= 20
    
    pdf.setFont("Helvetica", 10)
    
    with telegram_lock:
        subscribers = TELEGRAM_SUBSCRIBERS.copy()
    
    if not subscribers:
        pdf.setFillColor(colors.HexColor('#6b7280'))
        pdf.drawString(60, y_position, "No subscribers configured")
    else:
        for i, subscriber in enumerate(subscribers):
            if y_position < 100:  # Add new page if running out of space
                pdf.showPage()
                y_position = height - 50
                pdf.setFont("Helvetica", 10)
            
            # Background for subscriber
            pdf.setFillColor(colors.HexColor('#f9fafb'))
            pdf.rect(55, y_position - 15, width - 110, 18, fill=1, stroke=0)
            
            pdf.setFillColor(colors.HexColor('#1f2937'))
            name = subscriber.get('name', 'Unknown User')
            chat_id = subscriber.get('chat_id', 'N/A')
            username = subscriber.get('username', '')
            
            pdf.drawString(60, y_position - 10, f"• {name}")
            
            pdf.setFont("Helvetica", 9)
            pdf.setFillColor(colors.HexColor('#6b7280'))
            username_text = f"@{username}" if username else "(No username)"
            pdf.drawString(200, y_position - 10, f"ID: {chat_id} {username_text}")
            
            pdf.setFont("Helvetica", 10)
            y_position -= 20
    
    # Footer
    pdf.setFont("Helvetica", 9)
    pdf.setFillColor(colors.HexColor('#9ca3af'))
    pdf.drawString(50, 30, f"This is an official FallGuard incident report. Report ID: {incident['id']}")
    pdf.drawString(50, 15, "For more information, visit the FallGuard Admin Panel")
    
    pdf.save()
    
    buffer.seek(0)
    return Response(
        buffer.getvalue(),
        mimetype='application/pdf',
        headers={'Content-Disposition': f'attachment;filename=incident_{incident_id}.pdf'}
    )

@app.route('/api/incidents/<incident_id>/notes', methods=['POST'])
def api_update_incident_notes(incident_id):
    if not session.get('admin_authenticated', False):
        return jsonify({"success": False, "message": "Unauthorized"}), 401
    
    data = request.get_json() or {}
    notes = data.get('notes', '')
    
    incident = next((inc for inc in INCIDENT_REPORTS if inc['id'] == incident_id), None)
    if not incident:
        return jsonify({"success": False, "message": "Incident not found"}), 404
    
    incident['notes'] = notes
    save_incident_reports()
    
    return jsonify({"success": True, "message": "Notes updated"})

@app.route('/api/incidents/<incident_id>', methods=['DELETE'])
def api_delete_incident(incident_id):
    if not session.get('admin_authenticated', False):
        return jsonify({"success": False, "message": "Unauthorized"}), 401
    
    global INCIDENT_REPORTS
    INCIDENT_REPORTS = [inc for inc in INCIDENT_REPORTS if inc['id'] != incident_id]
    save_incident_reports()
    
    return jsonify({"success": True, "message": "Incident deleted"})

# Debug Routes
@app.route('/api/debug/cameras', methods=['GET'])
def api_debug_cameras():
    debug_info = {
        "definitions": {},
        "status": {},
        "shared_frames": list(shared_frames.keys()),
        "settings": GLOBAL_SETTINGS,
        "telegram": {
            "configured": TELEGRAM_BOT_TOKEN is not None,
            "subscribers": len(TELEGRAM_SUBSCRIBERS)
        },
        "frame_buffers": {cam_id: len(buf) for cam_id, buf in FRAME_BUFFERS.items()}
    }
    
    with camera_lock:
        for cam_id, cam_def in CAMERA_DEFINITIONS.items():
            processor = cam_def.get('thread_instance')
            debug_info["definitions"][cam_id] = {
                "name": cam_def['name'],
                "source": str(cam_def['source']),
                "has_processor": processor is not None,
                "is_running": processor.is_running if processor else False,
                "is_alive": processor.is_alive() if processor else False,
                "in_shared_frames": cam_id in shared_frames
            }
        
        for cam_id, status in CAMERA_STATUS.items():
            debug_info["status"][cam_id] = status
    
    return jsonify(debug_info)

# Startup
if __name__ == '__main__':
    print("\n" + "="*60)
    print("   FALLGUARD - AI Fall Detection System")
    print("="*60)
    
    # Load persistent data
    load_subscribers_from_file()
    load_blocked_list_from_file()
    load_incident_reports()
    
    # Auto-detect available cameras
    def detect_available_cameras():
        available_cameras = []
        print("\n[STARTUP] Scanning for available cameras...")
        
        # Test common camera indices
        for i in range(4):  # Check first 4 camera indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    available_cameras.append(i)
                    print(f"[CAMERA] Found camera at index {i}")
                cap.release()
            time.sleep(0.1)
        
        return available_cameras
    
    available_cams = detect_available_cameras()
    
    if available_cams:
        DEFAULT_CAMERA_SOURCE = available_cams[0]
        print(f"[STARTUP] Using camera index {DEFAULT_CAMERA_SOURCE} as default")
    else:
        DEFAULT_CAMERA_SOURCE = 0
        print(f"[STARTUP] No cameras detected, using default index 0")
    
    DEFAULT_CAMERA_ID = "main_webcam_0"
    DEFAULT_CAMERA_NAME = "Main Webcam"

    print(f"\n[STARTUP] Initializing default camera: {DEFAULT_CAMERA_NAME}")
    print(f"[INFO] Source: {DEFAULT_CAMERA_SOURCE}")
    print(f"[INFO] Model: {'LSTM' if LSTM_MODEL else 'Heuristic-based'}")
    print(f"[INFO] Detection: YOLOv11n-pose (20-30% faster, more accurate)")
    print(f"[INFO] Admin Password: {ADMIN_PASSWORD}")
    print(f"[INFO] Telegram: Bot listener started")
    print(f"[INFO] Privacy Modes: Enabled")
    print(f"[INFO] Pre-fall Buffer: {GLOBAL_SETTINGS['pre_fall_buffer_seconds']}s")
    print(f"[INFO] Incident Reports: Enabled")
    
    default_processor = CameraProcessor(
        camera_id=DEFAULT_CAMERA_ID, 
        src=DEFAULT_CAMERA_SOURCE, 
        name=DEFAULT_CAMERA_NAME,
        device=device
    )
    default_processor.start()

    with camera_lock:
        CAMERA_DEFINITIONS[DEFAULT_CAMERA_ID] = {
            "name": DEFAULT_CAMERA_NAME,
            "source": DEFAULT_CAMERA_SOURCE,
            "isLive": True,
            "thread_instance": default_processor
        }

    print(f"\n[INFO] Server starting on http://0.0.0.0:5000")
    print(f"[INFO] Access the system at: http://localhost:5000")
    print(f"[INFO] Admin panel: http://localhost:5000  Click 'Admin Panel'")
    print("="*60 + "\n")
    
    # Use Waitress WSGI server for proper multi-threading support
    sys.stdout.flush()
    from waitress import serve
    print("[INFO] Starting Waitress WSGI server with 4 worker threads...")
    sys.stdout.flush()
    serve(app, host='0.0.0.0', port=5000, threads=4)