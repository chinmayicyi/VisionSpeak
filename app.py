from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import cv2
import time
import os
from datetime import datetime
from ultralytics import YOLO
import threading
from threading import Lock
import pyttsx3
import math
from collections import defaultdict
import queue

app = Flask(__name__)
CORS(app)

# Global state
detector = None
detector_lock = Lock()
is_running = False
frame_lock = Lock()
current_frame = None

# Directories
SNAPSHOT_DIR = 'snapshots'
UPLOAD_DIR = 'uploads'
for directory in [SNAPSHOT_DIR, UPLOAD_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# ========================================
# Optimized Detector with TTS Queue
# ========================================

class OptimizedDetector:
    def __init__(self, camera_index=0, source_type='camera', source_path=None):
        print("Loading YOLO models...")
        
        # Load models
        self.models = {
            'general': YOLO('yolov8s.pt'),
        }
        
        print("Models loaded!")
        
        # Source setup
        self.source_type = source_type
        self.source_path = source_path
        
        if source_type == 'camera':
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                raise RuntimeError("Could not open camera")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        elif source_type == 'image':
            self.cap = None
            self.image = cv2.imread(source_path)
            if self.image is None:
                raise RuntimeError(f"Could not load image: {source_path}")
        elif source_type == 'video':
            self.cap = cv2.VideoCapture(source_path)
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open video: {source_path}")
        
        self.running = False
        self.narration_paused = False
        
        # TTS Queue System (CRITICAL FIX)
        self.tts_queue = queue.Queue()
        self.speaking = False
        self.tts_thread = None
        
        # Detection state
        self.last_objects = {}  # {name: count}
        self.last_narration_time = 0
        self.min_interval = 3.0
        self.periodic_interval = 8.0
        
        # Movement tracking
        self.tracked_positions = {}  # {name: [(x, y), timestamp]}
        self.movement_threshold = 60
        
        # Performance optimization
        self.frame_skip = 2  # Process every 2nd frame for detection
        self.frame_count = 0
        
        # Settings
        self.confidence_threshold = 0.35
        self.min_area = 2000
        
        # Start TTS worker thread
        self.start_tts_worker()
    
    def start_tts_worker(self):
        """Background thread that processes TTS queue"""
        def tts_worker():
            print("🔊 TTS worker started")
            
            while self.running or not self.tts_queue.empty():
                try:
                    # Get next message from queue (wait max 1 second)
                    text = self.tts_queue.get(timeout=1.0)
                    
                    if text:
                        self.speaking = True
                        print(f"[🔊 Speaking]: {text}")
                        
                        try:
                            # Create fresh engine for Windows compatibility
                            engine = pyttsx3.init()
                            
                            # Set female voice (CRITICAL FIX)
                            voices = engine.getProperty('voices')
                            if len(voices) > 1:
                                # Usually index 1 is female on Windows
                                engine.setProperty('voice', voices[1].id)
                            
                            engine.setProperty('rate', 150)
                            engine.setProperty('volume', 0.9)
                            
                            engine.say(text)
                            engine.runAndWait()
                            engine.stop()
                            del engine
                            
                        except Exception as e:
                            print(f"TTS Error: {e}")
                        
                        self.speaking = False
                        self.tts_queue.task_done()
                        time.sleep(0.3)  # Pause between speeches
                
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"TTS worker error: {e}")
                    self.speaking = False
            
            print("🔊 TTS worker stopped")
        
        self.tts_thread = threading.Thread(target=tts_worker, daemon=True)
        self.tts_thread.start()
    
    def queue_speech(self, text):
        """Add text to TTS queue (non-blocking)"""
        if not text or self.narration_paused:
            return
        
        # Don't queue if already in queue
        if self.tts_queue.qsize() < 3:  # Max 3 queued messages
            self.tts_queue.put(text)
            print(f"📝 Queued: {text}")
    
    def detect_movement(self, current_detections):
        """Detect object movement and direction"""
        current_time = time.time()
        movements = []
        
        current_positions = {}
        for det in current_detections:
            name = det['name']
            center = det['center']
            
            if name not in current_positions:
                current_positions[name] = []
            current_positions[name].append(center)
        
        # Check for movement
        for name, positions in current_positions.items():
            if name in self.tracked_positions:
                prev_positions, last_time = self.tracked_positions[name]
                
                if len(prev_positions) > 0 and len(positions) > 0:
                    # Average position
                    prev_x = sum(p[0] for p in prev_positions) / len(prev_positions)
                    prev_y = sum(p[1] for p in prev_positions) / len(prev_positions)
                    curr_x = sum(p[0] for p in positions) / len(positions)
                    curr_y = sum(p[1] for p in positions) / len(positions)
                    
                    dx = curr_x - prev_x
                    dy = curr_y - prev_y
                    distance = math.sqrt(dx*dx + dy*dy)
                    
                    if distance > self.movement_threshold and (current_time - last_time) > 1.0:
                        # Determine direction
                        if abs(dx) > abs(dy):
                            direction = "right" if dx > 0 else "left"
                        else:
                            direction = "down" if dy > 0 else "up"
                        
                        movements.append(f"The {name} is moving {direction}")
                        print(f"🏃 Movement: {name} moved {distance:.0f}px {direction}")
        
        # Update tracked positions
        for name, positions in current_positions.items():
            self.tracked_positions[name] = (positions, current_time)
        
        # Clean up old positions
        for name in list(self.tracked_positions.keys()):
            if name not in current_positions:
                del self.tracked_positions[name]
        
        return movements
    
    def create_natural_description(self, detections, movements=None):
        """Create natural language description with aggregation (CRITICAL FIX)"""
        
        # Priority for movements
        if movements:
            return movements[0]  # Announce first movement
        
        if not detections:
            return "No objects in view"
        
        # Count objects by type
        object_counts = defaultdict(int)
        for det in detections:
            object_counts[det['name']] += 1
        
        # Build natural sentence
        parts = []
        for name, count in sorted(object_counts.items()):
            if name == 'person':
                if count == 1:
                    parts.append("a person")
                else:
                    parts.append(f"{count} persons")
            else:
                if count == 1:
                    parts.append(f"a {name}")
                else:
                    parts.append(f"{count} {name}s")
        
        # Construct sentence
        if len(parts) == 0:
            return "No objects in view"
        elif len(parts) == 1:
            return f"I see {parts[0]}"
        elif len(parts) == 2:
            return f"I see {parts[0]} and {parts[1]}"
        elif len(parts) <= 4:
            return f"I see {', '.join(parts[:-1])}, and {parts[-1]}"
        else:
            total = sum(object_counts.values())
            return f"I see {total} objects including {parts[0]}, {parts[1]}, and others"
    
    def should_narrate(self, current_objects):
        """Decide when to narrate"""
        current_time = time.time()
        time_since_last = current_time - self.last_narration_time
        
        # Check if objects changed
        if current_objects != self.last_objects:
            if time_since_last > self.min_interval:
                return True, "change"
        
        # Periodic narration
        if time_since_last > self.periodic_interval:
            return True, "periodic"
        
        return False, None
    
    def process_frame(self, frame):
        """Process single frame with optimizations"""
        self.frame_count += 1
        
        # PERFORMANCE FIX: Skip detection on some frames
        if self.frame_count % self.frame_skip != 0:
            return frame, []
        
        all_detections = []
        
        for model_name, model in self.models.items():
            try:
                results = model(frame, conf=self.confidence_threshold, verbose=False)
                
                for r in results:
                    boxes = r.boxes
                    if boxes is not None:
                        for box in boxes:
                            conf = box.conf[0].item()
                            cls_id = int(box.cls[0].item())
                            class_name = r.names[cls_id]
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            
                            area = (x2 - x1) * (y2 - y1)
                            if area < self.min_area:
                                continue
                            
                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2
                            
                            detection = {
                                'name': class_name,
                                'confidence': conf,
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'center': (center_x, center_y),
                                'area': area
                            }
                            all_detections.append(detection)
            
            except Exception as e:
                print(f"Detection error: {e}")
                continue
        
        # Draw detections
        annotated_frame = self.draw_detections(frame.copy(), all_detections)
        
        # Check for narration
        if self.source_type == 'camera':  # Only narrate for live camera
            # Count objects
            object_counts = defaultdict(int)
            for det in all_detections:
                object_counts[det['name']] += 1
            
            # Check movement
            movements = self.detect_movement(all_detections)
            
            should_speak, reason = self.should_narrate(object_counts)
            
            if should_speak:
                description = self.create_natural_description(all_detections, movements)
                self.queue_speech(description)
                self.last_narration_time = time.time()
                self.last_objects = object_counts.copy()
        
        return annotated_frame, all_detections
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            name = det['name']
            conf = det['confidence']
            
            if name == 'person':
                color = (0, 255, 0)
            elif conf > 0.6:
                color = (255, 0, 255)
            else:
                color = (0, 255, 255)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"{name} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + 150, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Add stats overlay
        cv2.putText(frame, f"Objects: {len(detections)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return frame
    
    def run(self):
        """Main detection loop"""
        global current_frame
        
        print("🎥 Detection started!")
        
        if self.source_type == 'image':
            # Process single image
            annotated, detections = self.process_frame(self.image)
            with frame_lock:
                current_frame = annotated
            
            # Keep frame available
            while self.running:
                time.sleep(0.1)
        
        elif self.source_type in ['camera', 'video']:
            # Process video stream
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    if self.source_type == 'video':
                        # Loop video
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        break
                
                frame = cv2.flip(frame, 1) if self.source_type == 'camera' else frame
                
                annotated, detections = self.process_frame(frame)
                
                with frame_lock:
                    current_frame = annotated
                
                time.sleep(0.01)
        
        # Cleanup
        if self.cap:
            self.cap.release()
        print("Detection stopped!")

# ========================================
# Routes
# ========================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """MJPEG video stream"""
    def generate_frames():
        global current_frame, is_running
        
        while is_running:
            with frame_lock:
                if current_frame is not None:
                    _, buffer = cv2.imencode('.jpg', current_frame, 
                                            [cv2.IMWRITE_JPEG_QUALITY, 80])
                    frame_bytes = buffer.tobytes()
                    
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.033)  # ~30 FPS
    
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/start', methods=['POST'])
def api_start():
    """Start camera detection"""
    global detector, is_running
    
    if is_running:
        return jsonify({'status': 'already_running'})
    
    try:
        detector = OptimizedDetector(camera_index=0, source_type='camera')
        detector.running = True
        is_running = True
        
        thread = threading.Thread(target=detector.run, daemon=True)
        thread.start()
        
        return jsonify({'status': 'started'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def api_upload():
    """Upload and process image/video"""
    global detector, is_running
    
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected'}), 400
    
    # Save uploaded file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    ext = os.path.splitext(file.filename)[1]
    filename = f'upload_{timestamp}{ext}'
    filepath = os.path.join(UPLOAD_DIR, filename)
    file.save(filepath)
    
    # Determine type
    if ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
        source_type = 'image'
    elif ext.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        source_type = 'video'
    else:
        return jsonify({'status': 'error', 'message': 'Unsupported file type'}), 400
    
    try:
        # Stop existing detector
        if is_running and detector:
            detector.running = False
            time.sleep(0.5)
        
        # Start new detector with uploaded file
        detector = OptimizedDetector(source_type=source_type, source_path=filepath)
        detector.running = True
        is_running = True
        
        thread = threading.Thread(target=detector.run, daemon=True)
        thread.start()
        
        return jsonify({'status': 'started', 'type': source_type, 'filename': filename})
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/stop', methods=['POST'])
def api_stop():
    """Stop detection"""
    global detector, is_running, current_frame
    
    try:
        is_running = False
        if detector:
            detector.running = False
            time.sleep(0.5)
            detector = None
        
        with frame_lock:
            current_frame = None
        
        return jsonify({'status': 'stopped'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/status')
def api_status():
    """Get detection status"""
    global detector, is_running
    
    if not is_running or not detector:
        return jsonify({'running': False, 'objects': [], 'object_count': 0})
    
    object_counts = detector.last_objects
    objects = [{'name': name, 'count': count} for name, count in object_counts.items()]
    total_count = sum(object_counts.values())
    
    return jsonify({
        'running': True,
        'objects': objects,
        'object_count': total_count,
        'narration_paused': detector.narration_paused,
        'source_type': detector.source_type
    })

@app.route('/api/toggle_narration', methods=['POST'])
def api_toggle_narration():
    """Toggle narration"""
    global detector
    
    if detector:
        detector.narration_paused = not detector.narration_paused
        status = 'paused' if detector.narration_paused else 'active'
        return jsonify({'status': status})
    
    return jsonify({'status': 'error', 'message': 'Detector not running'}), 400

@app.route('/api/force_speak', methods=['POST'])
def api_force_speak():
    """Force speak current scene"""
    global detector
    
    if detector and detector.last_objects:
        # Create description from current objects
        fake_detections = []
        for name, count in detector.last_objects.items():
            for _ in range(count):
                fake_detections.append({'name': name})
        
        description = detector.create_natural_description(fake_detections)
        detector.queue_speech(description)
        return jsonify({'status': 'speaking', 'text': description})
    
    return jsonify({'status': 'error', 'message': 'No objects to speak'}), 400

@app.route('/api/snapshot', methods=['POST'])
def api_snapshot():
    """Save snapshot"""
    global current_frame
    
    with frame_lock:
        if current_frame is not None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'snapshot_{timestamp}.jpg'
            filepath = os.path.join(SNAPSHOT_DIR, filename)
            cv2.imwrite(filepath, current_frame)
            return jsonify({'success': True, 'filename': filename})
    
    return jsonify({'success': False, 'error': 'No frame available'}), 400

@app.route('/api/change_confidence', methods=['POST'])
def api_change_confidence():
    """Change confidence threshold"""
    global detector
    
    data = request.get_json()
    threshold = data.get('threshold', 0.35)
    threshold = max(0.1, min(0.9, threshold))
    
    if detector:
        detector.confidence_threshold = threshold
        return jsonify({'threshold': threshold})
    
    return jsonify({'status': 'error'}), 400

if __name__ == '__main__':
    print("="*60)
    print("🎥 VisionSpeak Optimized Backend")
    print("="*60)
    print("Server: http://localhost:5000")
    print("="*60)
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)