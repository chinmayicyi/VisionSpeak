from flask import Flask, render_template, Response, jsonify, request, send_file
from flask_cors import CORS
import cv2
import time
import os
from datetime import datetime
from flash import MultiModelDetector
import threading
from werkzeug.utils import secure_filename
import queue

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
SNAPSHOT_DIR = 'snapshots'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}

for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, SNAPSHOT_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max

# Global state
detector = None
detector_lock = threading.Lock()
is_running = False
current_processed_file = None

# ========================================
# Helper Functions
# ========================================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_video_file(filename):
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in {'mp4', 'avi', 'mov'}

def is_image_file(filename):
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in {'png', 'jpg', 'jpeg'}

# ========================================
# Routes
# ========================================

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """
    MJPEG video stream endpoint for live camera
    """
    def generate_frames():
        global detector, is_running
        
        while is_running and detector:
            if hasattr(detector, 'current_frame') and detector.current_frame is not None:
                with detector.frame_lock:
                    frame_copy = detector.current_frame.copy()
                
                # Encode frame to JPEG
                _, buffer = cv2.imencode('.jpg', frame_copy, 
                                        [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_bytes = buffer.tobytes()
                
                # Yield frame in MJPEG format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.033)  # ~30 FPS
    
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

# ========================================
# API Endpoints - Live Camera
# ========================================

@app.route('/api/start', methods=['POST'])
def api_start():
    """Start live camera detection"""
    global detector, is_running
    
    if is_running:
        return jsonify({'status': 'already_running'})
    
    try:
        with detector_lock:
            detector = MultiModelDetector(camera_index=0, display_queue=None)
            detector.running = True
            is_running = True
        
        # Start detector in background thread
        thread = threading.Thread(target=detector.run, daemon=True)
        thread.start()
        
        return jsonify({'status': 'started'})
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/stop', methods=['POST'])
def api_stop():
    """Stop live camera detection"""
    global detector, is_running
    
    try:
        with detector_lock:
            is_running = False
            if detector:
                detector.running = False
                if detector.cap:
                    detector.cap.release()
                detector = None
        
        return jsonify({'status': 'stopped'})
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/status')
def api_status():
    """Get current detection status"""
    global detector, is_running
    
    if not is_running or not detector:
        return jsonify({
            'running': False,
            'objects': [],
            'object_count': 0
        })
    
    objects = []
    object_count = 0
    
    if hasattr(detector, 'last_objects'):
        objects = [{'name': obj} for obj in detector.last_objects]
        object_count = len(detector.last_objects)
    
    return jsonify({
        'running': True,
        'objects': objects,
        'object_count': object_count,
        'narration_paused': getattr(detector, 'narration_paused', False)
    })

@app.route('/api/toggle_narration', methods=['POST'])
def api_toggle_narration():
    """Toggle narration on/off"""
    global detector
    
    if detector:
        detector.narration_paused = not detector.narration_paused
        status = 'paused' if detector.narration_paused else 'active'
        return jsonify({'status': status})
    
    return jsonify({'status': 'error', 'message': 'Detector not running'}), 400

@app.route('/api/force_speak', methods=['POST'])
def api_force_speak():
    """Force immediate narration"""
    global detector
    
    if detector:
        detector.force_narrate()
        return jsonify({'status': 'speaking'})
    
    return jsonify({'status': 'error', 'message': 'Detector not running'}), 400

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
    
    return jsonify({'status': 'error', 'message': 'Detector not running'}), 400

@app.route('/api/snapshot', methods=['POST'])
def api_snapshot():
    """Save current frame as snapshot"""
    global detector
    
    if detector and hasattr(detector, 'current_frame') and detector.current_frame is not None:
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'snapshot_{timestamp}.jpg'
            filepath = os.path.join(SNAPSHOT_DIR, filename)
            
            with detector.frame_lock:
                cv2.imwrite(filepath, detector.current_frame)
            
            return jsonify({
                'success': True,
                'filename': filename,
                'path': filepath
            })
        
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    return jsonify({'success': False, 'error': 'No frame available'}), 400

# ========================================
# API Endpoints - Media Upload
# ========================================

@app.route('/api/upload', methods=['POST'])
def api_upload():
    """Upload and process image or video"""
    global current_processed_file
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process file
        if is_image_file(filename):
            processed_path = process_image(filepath)
        else:
            processed_path = process_video(filepath)
        
        current_processed_file = processed_path
        
        return jsonify({
            'success': True,
            'filename': os.path.basename(processed_path),
            'file_type': 'image' if is_image_file(filename) else 'video',
            'url': f'/processed/{os.path.basename(processed_path)}'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/processed/<filename>')
def serve_processed(filename):
    """Serve processed media file"""
    return send_file(os.path.join(PROCESSED_FOLDER, filename))

def process_image(image_path):
    """Process a single image with YOLO detection"""
    from flash import MultiModelDetector
    
    # Load image
    frame = cv2.imread(image_path)
    
    # Create temporary detector
    temp_detector = MultiModelDetector(camera_index=None, display_queue=None)
    
    # Run detection
    detections = temp_detector._run_detection(frame)
    annotated_frame = temp_detector._draw_detections(frame, detections)
    
    # Narrate what was found
    if detections:
        description = temp_detector._create_description(detections)
        temp_detector._speak_async(description)
    else:
        temp_detector._speak_async("No objects detected in this image")
    
    # Save processed image
    output_filename = f"processed_{os.path.basename(image_path)}"
    output_path = os.path.join(PROCESSED_FOLDER, output_filename)
    cv2.imwrite(output_path, annotated_frame)
    
    return output_path

def process_video(video_path):
    """Process a video file with YOLO detection"""
    from flash import MultiModelDetector
    
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output video writer
    output_filename = f"processed_{os.path.basename(video_path)}"
    output_path = os.path.join(PROCESSED_FOLDER, output_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create temporary detector
    temp_detector = MultiModelDetector(camera_index=None, display_queue=None)
    
    # Process video frame by frame
    frame_count = 0
    last_narration = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection every 3rd frame for performance
        if frame_count % 3 == 0:
            detections = temp_detector._run_detection(frame)
            annotated_frame = temp_detector._draw_detections(frame, detections)
            
            # Narrate occasionally (every 3 seconds)
            if detections and (time.time() - last_narration) > 3:
                description = temp_detector._create_description(detections)
                temp_detector._speak_async(description)
                last_narration = time.time()
        else:
            annotated_frame = frame
        
        out.write(annotated_frame)
        frame_count += 1
    
    cap.release()
    out.release()
    
    return output_path

# ========================================
# Health Check
# ========================================

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'detector_running': is_running
    })

# ========================================
# Main
# ========================================

if __name__ == '__main__':
    print("="*60)
    print("🎥 VisionSpeak Backend Server (Optimized)")
    print("="*60)
    print("Starting server on http://localhost:5000")
    print("Open browser to http://localhost:5000")
    print("="*60)
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)