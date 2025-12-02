from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import cv2
import time
import os
from datetime import datetime
from flash import MultiModelDetector
import threading

app = Flask(__name__)
CORS(app)

# Global state
detector = None
detector_lock = threading.Lock()
is_running = False

# Snapshot directory
SNAPSHOT_DIR = 'snapshots'
if not os.path.exists(SNAPSHOT_DIR):
    os.makedirs(SNAPSHOT_DIR)

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
    MJPEG video stream endpoint
    This is what the frontend uses for live video
    """
    def generate_frames():
        global detector, is_running
        
        while is_running and detector:
            # Get frame from detector
            if hasattr(detector, 'current_frame') and detector.current_frame is not None:
                # Encode frame to JPEG
                _, buffer = cv2.imencode('.jpg', detector.current_frame, 
                                        [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_bytes = buffer.tobytes()
                
                # Yield frame in MJPEG format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.033)  # ~30 FPS
    
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

# ========================================
# API Endpoints
# ========================================

@app.route('/api/start', methods=['POST'])
def api_start():
    """Start detection"""
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
    """Stop detection"""
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
    
    # Get current detections (you need to store these in detector)
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
    
    if detector and not detector.speaking:
        # Get current objects
        current_objects = list(getattr(detector, 'last_objects', set()))
        
        if current_objects:
            # Create description
            if len(current_objects) == 1:
                text = f"I can see a {current_objects[0]}"
            elif len(current_objects) == 2:
                text = f"I can see a {current_objects[0]} and a {current_objects[1]}"
            else:
                text = f"I can see {', '.join(current_objects[:-1])}, and a {current_objects[-1]}"
            
            # Speak
            detector._speak_async(text)
            return jsonify({'status': 'speaking', 'text': text})
        else:
            detector._speak_async("No objects currently detected")
            return jsonify({'status': 'speaking', 'text': 'No objects'})
    
    return jsonify({'status': 'error', 'message': 'Cannot speak'}), 400

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
            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'snapshot_{timestamp}.jpg'
            filepath = os.path.join(SNAPSHOT_DIR, filename)
            
            # Save frame
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
    print("🎥 VisionSpeak Backend Server")
    print("="*60)
    print("Starting server on http://localhost:5000")
    print("Open browser to http://localhost:5000")
    print("="*60)
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)