import os
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

# --- IMPORT MODULES ---
import torch
from app.skeleton_lstm import LSTMModel, SEQUENCE_LENGTH, FEATURE_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE
from app.video_utils import extract_55_features, draw_skeleton, predict_torch 
import mediapipe as mp

# --- Global Settings ---
DEFAULT_FALL_THRESHOLD = 0.70
INTERNAL_FPS = 30
DEFAULT_FALL_DELAY_SECONDS = 2
DEFAULT_ALERT_COOLDOWN_SECONDS = 60

GLOBAL_SETTINGS = {
    "fall_threshold": DEFAULT_FALL_THRESHOLD,
    "fall_delay_seconds": DEFAULT_FALL_DELAY_SECONDS,
    "alert_cooldown_seconds": DEFAULT_ALERT_COOLDOWN_SECONDS
}

# Telegram Settings
TELEGRAM_BOT_TOKEN = "8204879198:AAErRTPpGXDZGsXO7ZoF9VtTWbDJB9isxzA"
TELEGRAM_SUBSCRIBERS = []  # List of {chat_id, name, username}
TELEGRAM_SUBSCRIBERS_FILE = "data/telegram_subscribers.json"
TELEGRAM_BLOCKED_LIST = []  # Chat IDs that have been removed and should stay removed
TELEGRAM_BLOCKED_LIST_FILE = "data/telegram_blocked_list.json"
telegram_lock = threading.Lock()

# Admin Password
ADMIN_PASSWORD = "admin"

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
            print(f"           Features: {FEATURE_SIZE}, Hidden: {HIDDEN_SIZE}, Output: {OUTPUT_SIZE}")
        else:
            print(f"[WARNING] Model file not found: {MODEL_FILE}")
            LSTM_MODEL = None
    except RuntimeError as e:
        if "size mismatch" in str(e):
            print(f"[INFO] Model shape mismatch detected, trying legacy format (output_size=1)")
            LSTM_MODEL = LSTMModel(FEATURE_SIZE, HIDDEN_SIZE, 1, NUM_LAYERS)
            
            if os.path.exists(MODEL_FILE):
                LSTM_MODEL.load_state_dict(torch.load(MODEL_FILE, map_location=device, weights_only=True))
                LSTM_MODEL.to(device)
                LSTM_MODEL.eval()
                print(f"[SUCCESS] Loaded LSTM Model (legacy format) from {MODEL_FILE}")
            else:
                LSTM_MODEL = None
        else:
            raise
except Exception as e:
    print(f"[ERROR] Failed to load LSTM model: {e}") 
    print(f"[INFO] Falling back to enhanced heuristic detection")
    LSTM_MODEL = None

# MediaPipe Setup
USE_MEDIAPIPE = False
try:
    mp_pose = mp.solutions.pose
    USE_MEDIAPIPE = True
    print("[SUCCESS] MediaPipe initialized successfully")
except Exception as e:
    print(f"[ERROR] MediaPipe not available: {e}")

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

# Flask app 
app = Flask(__name__, static_folder='app', static_url_path='')
app.secret_key = os.environ.get("FALLGUARD_SECRET", "fallguard_secret_key_2024")
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
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
        # Use offset 0 to get all updates from the start, with a high limit
        params = {'offset': 0, 'limit': 1000, 'timeout': 10}
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            updates = data.get('result', [])
            loaded_ids = set()
            
            # Process all updates in reverse order (oldest first)
            for update in reversed(updates):
                message = update.get('message', {})
                text = message.get('text', '')
                chat_id = message.get('chat', {}).get('id')
                username = message.get('chat', {}).get('username', '')
                first_name = message.get('chat', {}).get('first_name', '')
                last_name = message.get('chat', {}).get('last_name', '')
                
                # Look for /start commands
                if text.startswith('/start') and chat_id and chat_id not in loaded_ids:
                    with telegram_lock:
                        # Check if not already in list
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
    
    # On first run, load all previous subscribers from Telegram history
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
                            # Check if blocked
                            if str(chat_id) in TELEGRAM_BLOCKED_LIST:
                                send_telegram_message(
                                    chat_id,
                                    "❌ You have been removed from FallGuard alerts."
                                )
                                print(f"[TELEGRAM] Blocked user tried /start: ID {chat_id}")
                                continue
                            
                            # Check if already subscribed
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
        self.last_alert_time = 0  # Track last alert time for cooldown
        
        self.mp_pose_instance = None
        if USE_MEDIAPIPE:
            self.mp_pose_instance = mp_pose.Pose(
                static_image_mode=False, 
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                enable_segmentation=False,
                smooth_landmarks=True
            ) 
        
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

    def update_fall_timer_threshold(self):
        delay_seconds = GLOBAL_SETTINGS['fall_delay_seconds']
        frame_threshold = max(1, round(delay_seconds * INTERNAL_FPS)) 
        self.fall_timer = FallTimer(threshold_frames=frame_threshold)

    def update_camera_status(self, status, color, last_alert=None, is_live=True):
        with camera_lock:
            if self.camera_id not in CAMERA_STATUS:
                CAMERA_STATUS[self.camera_id] = {}
            
            CAMERA_STATUS[self.camera_id].update({
                "status": status,
                "color": color,
                "isLive": is_live,
                "name": self.name,
                "source": str(self.src),
                "confidence_score": self.latest_fall_prob,
                "model_threshold": GLOBAL_SETTINGS['fall_threshold'],
                "fps": self.current_fps
            })
            if last_alert:
                CAMERA_STATUS[self.camera_id]["lastAlert"] = time.ctime(last_alert)

    def send_fall_alert(self, frame):
        """Send fall alert via Telegram with snapshot"""
        current_time = time.time()
        cooldown = GLOBAL_SETTINGS['alert_cooldown_seconds']
        
        # Check cooldown
        if current_time - self.last_alert_time < cooldown:
            return
        
        self.last_alert_time = current_time
        
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_SUBSCRIBERS:
            print(f"[TELEGRAM] No bot token or subscribers configured")
            return
        
        # Prepare alert message
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        confidence = self.latest_fall_prob * 100
        
        caption = (
            f"🚨 <b>FALL DETECTED</b> 🚨\n\n"
            f"📍 <b>Location:</b> {self.name}\n"
            f"🕐 <b>Time:</b> {timestamp}\n"
            f"📊 <b>Confidence:</b> {confidence:.1f}%\n\n"
            f"⚠️ Please check on the person immediately!"
        )
        
        # Encode frame as JPEG
        try:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            ret, jpeg = cv2.imencode('.jpg', frame, encode_param)
            if not ret:
                return
            
            photo_bytes = jpeg.tobytes()
            
            # Send to all subscribers
            with telegram_lock:
                for subscriber in TELEGRAM_SUBSCRIBERS:
                    try:
                        success = send_telegram_photo(subscriber['chat_id'], photo_bytes, caption)
                        if success:
                            print(f"[TELEGRAM] Alert sent to {subscriber['name']}")
                    except Exception as e:
                        print(f"[ERROR] Failed to send alert to {subscriber['name']}: {e}")
        
        except Exception as e:
            print(f"[ERROR] Failed to prepare fall alert: {e}")

    def extract_features_and_bbox(self, frame):
        if USE_MEDIAPIPE and self.mp_pose_instance:
            try:
                bbox, feature_vec, pose_results = extract_55_features(frame, self.mp_pose_instance)
                self.latest_pose_results = pose_results
                self.latest_features = feature_vec
                
                if feature_vec is not None and np.any(feature_vec != 0):
                    pass
                else:
                    feature_vec = np.zeros(FEATURE_SIZE, dtype=np.float32)
                    bbox = None
                    pose_results = None
            except Exception as e:
                print(f"[ERROR] Feature extraction failed: {e}")
                feature_vec = np.zeros(FEATURE_SIZE, dtype=np.float32)
                bbox = None
                pose_results = None
        else:
            feature_vec = np.zeros(FEATURE_SIZE, dtype=np.float32)
            bbox = None
            pose_results = None
            
        self.pose_sequence.append(feature_vec)
        return bbox, feature_vec, pose_results

    def predict_fall_enhanced(self):
        current_threshold = GLOBAL_SETTINGS['fall_threshold']
        fall_probability = 0.0

        if len(self.pose_sequence) > 0 and self.latest_features is not None:
            features = self.latest_features
            
            HWR = features[0]
            TorsoAngle = features[1]
            D = features[2]
            H = features[5]
            FallAngleD = features[6]
            
            fall_score = 0.0
            
            if 0.0 < HWR < 0.7:
                fall_score += 0.3
                if HWR < 0.5:
                    fall_score += 0.2
            
            if TorsoAngle > 45:
                fall_score += 0.25
                if TorsoAngle > 70:
                    fall_score += 0.15
            
            if H > 0.6:
                fall_score += 0.2
                if H > 0.75:
                    fall_score += 0.2
            
            if FallAngleD < 30:
                fall_score += 0.3
            
            if abs(D) < 0.15:
                fall_score += 0.15
            
            fall_probability = min(fall_score, 0.99)
            
            if hasattr(self, 'debug_counter'):
                self.debug_counter += 1
            else:
                self.debug_counter = 0
                
            if self.debug_counter % 30 == 0:
                print(f"[{self.name}] Heuristic: HWR={HWR:.2f}, Torso={TorsoAngle:.0f}°, H={H:.2f}, Angle={FallAngleD:.0f}°, Score={fall_probability:.2f}")

        if LSTM_MODEL is not None and len(self.pose_sequence) >= self.sequence_length:
            try:
                input_data = np.array(self.pose_sequence, dtype=np.float32)
                input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    pred, prob = predict_torch(LSTM_MODEL, input_tensor, threshold=current_threshold)
                
                if prob > fall_probability:
                    fall_probability = prob
                    if self.debug_counter % 30 == 0:
                        print(f"[{self.name}] LSTM override: {prob:.2f}")
            except Exception as e:
                print(f"[ERROR] LSTM prediction failed for {self.camera_id}: {e}")

        self.latest_fall_prob = fall_probability
        return (fall_probability >= current_threshold), fall_probability

    def draw_enhanced_overlay(self, frame, bbox, fall_confirmed, fall_prob, current_threshold, feature_vector):
        h, w, _ = frame.shape
        
        if bbox is not None and bbox[2] > 0 and bbox[3] > 0:
            x, y, bw, bh = bbox
            color = (0, 0, 255) if fall_confirmed else (0, 255, 0)
            thickness = 5 if fall_confirmed else 2
            
            cv2.rectangle(frame, (int(x), int(y)), (int(x + bw), int(y + bh)), color, thickness)
            
            label = "FALL" if fall_confirmed else "PERSON"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            label_x = int(x + (bw - label_size[0]) / 2)
            label_y = max(30, int(y - 15))
            
            cv2.rectangle(frame, (label_x - 8, label_y - label_size[1] - 8), 
                         (label_x + label_size[0] + 8, label_y + 8), color, -1)
            cv2.putText(frame, label, (label_x, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        
        return frame

    def run(self):
        self.update_fall_timer_threshold()
        self.update_camera_status("Starting...", "gray", is_live=True)
        
        print(f"[{self.name}] Opening video source: {self.src}")
        
        max_retries = 3
        for attempt in range(max_retries):
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

        if not is_video_file:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        else:
            video_fps = self.cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"[{self.name}] Video file: FPS={video_fps}, Frames={total_frames}")

        ret, first_frame = self.cap.read()
        if ret:
            first_frame = cv2.resize(first_frame, (640, 480))
            with shared_frames[self.camera_id]["lock"]:
                shared_frames[self.camera_id]["frame"] = first_frame
            
            if is_video_file:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        self.is_running = True
        self.update_camera_status("Active", "green", is_live=True)
        
        print(f"[SUCCESS] Camera '{self.name}' started successfully")

        consecutive_failures = 0
        max_failures = 30
        
        try:
            while self.is_running:
                start_time = time.time()
                
                ret, frame = self.cap.read()
                
                if is_video_file and not ret:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
                    ret, frame = self.cap.read()

                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        print(f"[ERROR] {self.name}: Too many consecutive failures")
                        break
                    time.sleep(0.05)
                    continue
                
                consecutive_failures = 0
                
                frame = cv2.resize(frame, (640, 480))
                
                try:
                    bbox, feature_vec, pose_results = self.extract_features_and_bbox(frame)
                except Exception as e:
                    bbox, feature_vec, pose_results = None, None, None

                is_falling, fall_prob = self.predict_fall_enhanced()
                fall_confirmed = self.fall_timer.update(is_falling) 
                current_threshold = GLOBAL_SETTINGS['fall_threshold']

                if fall_confirmed:
                    self.update_camera_status("FALL DETECTED", "red", last_alert=time.time())
                    # Send Telegram alert
                    self.send_fall_alert(frame.copy())
                elif is_falling:
                    self.update_camera_status("Analyzing", "yellow")
                else:
                    self.update_camera_status("Normal", "green")

                processed = frame.copy()
                
                if self.latest_pose_results:
                    processed = draw_skeleton(processed, self.latest_pose_results, fall_confirmed)

                processed = self.draw_enhanced_overlay(processed, bbox, fall_confirmed, 
                                                      fall_prob, current_threshold, feature_vec)

                with shared_frames[self.camera_id]["lock"]:
                    shared_frames[self.camera_id]["frame"] = processed

                self.frame_count += 1
                if time.time() - self.last_fps_update >= 1.0:
                    self.current_fps = self.frame_count / (time.time() - self.last_fps_update)
                    self.frame_count = 0
                    self.last_fps_update = time.time()

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
            import traceback
            traceback.print_exc()
        finally:
            if self.mp_pose_instance: 
                self.mp_pose_instance.close()
            if self.cap: 
                self.cap.release()
            
            with camera_lock:
                if self.camera_id in CAMERA_STATUS: 
                    del CAMERA_STATUS[self.camera_id]
                if self.camera_id in CAMERA_DEFINITIONS: 
                    CAMERA_DEFINITIONS[self.camera_id]['isLive'] = False
                    CAMERA_DEFINITIONS[self.camera_id]['thread_instance'] = None
                
            print(f"[INFO] Camera '{self.name}' stopped")

# MJPEG Stream Generator
def generate_mjpeg(camera_id):
    boundary = b'--frame\r\n'
    
    wait_time = 0
    max_wait = 5
    
    while camera_id not in shared_frames and wait_time < max_wait:
        time.sleep(0.1)
        wait_time += 0.1
    
    if camera_id not in shared_frames:
        placeholder = 100 * np.ones((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Camera Not Available", (150, 220), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(placeholder, f"ID: {camera_id}", (200, 260), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
        
        ret, jpeg = cv2.imencode('.jpg', placeholder)
        frame_bytes = jpeg.tobytes()
        yield boundary + b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        return
        
    while camera_id in shared_frames:
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
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            ret, jpeg = cv2.imencode('.jpg', frame, encode_param)
            frame_bytes = jpeg.tobytes()

        yield boundary + b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        time.sleep(0.033)

# Flask Routes
@app.route('/')
def index():
    return send_from_directory('app', 'index.html')

@app.route('/video_feed/<camera_id>')
def video_feed(camera_id):
    return Response(generate_mjpeg(camera_id), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

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
    if not session.get('admin_authenticated', False):
        return jsonify({"success": False, "message": "Unauthorized"}), 401
    
    if request.method == 'POST':
        data = request.get_json() or {}
        message = []
        
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

        return jsonify({"success": True, "message": " ".join(message), "settings": GLOBAL_SETTINGS})
    
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

@app.route('/api/cameras', methods=['GET'])
def api_get_cameras():
    cameras = []
    with camera_lock:
        for cam_id, cam_def in CAMERA_DEFINITIONS.items():
            processor = cam_def.get('thread_instance')
            
            is_actually_live = False
            if processor is not None:
                try:
                    is_actually_live = processor.is_running and processor.is_alive()
                except:
                    is_actually_live = False
            
            status = CAMERA_STATUS.get(cam_id, {
                "status": "Offline", 
                "color": "gray", 
                "isLive": False, 
                "confidence_score": 0.0,
                "fps": 0
            })
            
            actual_is_live = is_actually_live and cam_id in shared_frames
            
            cameras.append({
                "id": cam_id,
                "name": cam_def['name'],
                "source": str(cam_def['source']),
                "isLive": actual_is_live,
                "status": status['status'] if actual_is_live else "Offline",
                "color": status['color'] if actual_is_live else "gray",
                "lastAlert": status.get('lastAlert', 'N/A'),
                "confidence_score": status.get('confidence_score', 0.0),
                "model_threshold": GLOBAL_SETTINGS['fall_threshold'],
                "fps": status.get('fps', 0)
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
def api_upload_camera():
    if 'video_file' not in request.files:
        return jsonify({"success": False, "message": "No file uploaded"}), 400
    
    file = request.files['video_file']
    name = request.form.get('name')

    if file.filename == '':
        return jsonify({"success": False, "message": "No file selected"}), 400
    if not name:
        return jsonify({"success": False, "message": "Camera name required"}), 400
    
    allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        return jsonify({"success": False, "message": f"Unsupported file type: {file_ext}"}), 400
    
    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4().hex}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    
    try:
        print(f"[UPLOAD] Saving file: {filename} -> {filepath}")
        file.save(filepath)
        print(f"[UPLOAD] File saved successfully: {filepath}")
        
        if not os.path.exists(filepath):
            return jsonify({"success": False, "message": "File save failed"}), 500
            
        file_size = os.path.getsize(filepath)
        print(f"[UPLOAD] File size: {file_size / (1024*1024):.2f} MB")
        
        test_cap = cv2.VideoCapture(filepath)
        if not test_cap.isOpened():
            test_cap.release()
            os.remove(filepath)
            return jsonify({"success": False, "message": "Invalid video file - cannot be read"}), 400
        
        ret, _ = test_cap.read()
        test_cap.release()
        
        if not ret:
            os.remove(filepath)
            return jsonify({"success": False, "message": "Video file is empty or corrupted"}), 400
        
    except Exception as e:
        print(f"[ERROR] Upload failed: {e}")
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass
        return jsonify({"success": False, "message": f"Upload failed: {str(e)}"}), 500

    camera_id = f"cam_{str(uuid.uuid4())[:8]}"
    name_safe = name

    print(f"[UPLOAD] Starting camera processor for: {name_safe} (ID: {camera_id})")
    
    processor = CameraProcessor(camera_id=camera_id, src=filepath, name=name_safe, device=device)
    processor.start()
    
    with camera_lock:
        CAMERA_DEFINITIONS[camera_id] = {
            "name": name_safe, 
            "source": filepath, 
            "isLive": True,
            "thread_instance": processor
        }
    
    print(f"[UPLOAD] Camera started: {name_safe}")
    
    return jsonify({
        "success": True, 
        "message": f"Video uploaded: {name_safe}", 
        "camera_id": camera_id,
        "file_size_mb": f"{file_size / (1024*1024):.2f}"
    })

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
        }
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
    
    # Load subscribers and blocked list from persistent storage
    load_subscribers_from_file()
    load_blocked_list_from_file()
    
    DEFAULT_CAMERA_ID = "main_webcam_0"
    DEFAULT_CAMERA_NAME = "Main Webcam"
    DEFAULT_CAMERA_SOURCE = 0

    print(f"\n[STARTUP] Initializing default camera: {DEFAULT_CAMERA_NAME}")
    print(f"[INFO] Source: {DEFAULT_CAMERA_SOURCE}")
    print(f"[INFO] Model: {'LSTM' if LSTM_MODEL else 'Heuristic-based'}")
    print(f"[INFO] MediaPipe: {'Enabled' if USE_MEDIAPIPE else 'Disabled'}")
    print(f"[INFO] Admin Password: {ADMIN_PASSWORD}")
    print(f"[INFO] Telegram: Bot listener started")
    
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
    print(f"[INFO] Debug endpoint: http://localhost:5000/api/debug/cameras")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)