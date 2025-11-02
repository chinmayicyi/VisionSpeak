import cv2
import threading
import time
from flask import Flask, render_template, Response
from flask_sock import Sock
from flash import MultiModelDetector 
try:
    from waitress import serve
except ImportError:
    print("Waitress not found. Please run: pip install waitress")
    exit()

# --- Initialize Flask App & WebSocket ---
app = Flask(__name__)
sock = Sock(app)

# --- Thread-Safe Global Variables ---
latest_raw_frame = None
raw_frame_lock = threading.Lock()

output_frame = None
output_frame_lock = threading.Lock()

websocket_clients = set()
ws_lock = threading.Lock()

# --- WebSocket Helper Functions ---

def register_client(ws):
    with ws_lock:
        print(f"Client connected: {ws}")
        websocket_clients.add(ws)

def unregister_client(ws):
    with ws_lock:
        print(f"Client disconnected: {ws}")
        websocket_clients.discard(ws)

def broadcast_narration(text):
    clients_to_remove = []
    with ws_lock:
        for client in set(websocket_clients):
            try:
                client.send(text)
            except Exception as e:
                print(f"Error sending to client {client}: {e}. Marking for removal.")
                clients_to_remove.append(client)
    
    if clients_to_remove:
        with ws_lock:
            for client in clients_to_remove:
                websocket_clients.discard(client)

# --- Background Thread 1: Camera Reader ---

def read_camera_thread():
    """
    This thread is NOW the ONLY place that touches the camera.
    """
    global latest_raw_frame, raw_frame_lock
    
    print("Camera thread started... opening camera.")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("CRITICAL: Camera thread could not open camera.")
        return
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Camera opened successfully.")
    
    while True:
        success, frame = cap.read()
        if not success:
            print("Camera thread: Failed to grab frame.")
            time.sleep(0.1)
            continue
        
        with raw_frame_lock:
            latest_raw_frame = frame.copy()

# --- Background Thread 2: Detection Processor ---

def run_detection_thread():
    """
    This thread NOW creates its own Detector instance.
    """
    global latest_raw_frame, raw_frame_lock
    global output_frame, output_frame_lock
    
    print("Detection thread started... loading models.")
    
    try:
        detector = MultiModelDetector()
        print("Detection thread loaded models successfully.")
    except Exception as e:
        print(f"CRITICAL: Detection thread failed to load models: {e}")
        return

    last_spoken_time = 0
    spoken_objects = set()

    while True:
        with raw_frame_lock:
            if latest_raw_frame is None:
                time.sleep(0.1)
                continue
            current_frame = latest_raw_frame.copy()
        
        try:
            detections = detector.detect_with_multiple_models(current_frame)
            detections = [d for d in detections if d['area'] > 1000]
            
            now = time.time()
            current_objects = set([d['name'] for d in detections])
            
            should_speak = False
            if now - last_spoken_time > 4: 
                should_speak = True
            elif current_objects != spoken_objects and current_objects: 
                should_speak = True
            
            # Also speak "No objects detected" if objects were just cleared
            if should_speak and (detections or spoken_objects):
                description = detector.create_description(detections)
                print(f"[Broadcasting]: {description}")
                broadcast_narration(description)
                spoken_objects = current_objects
                last_spoken_time = now
            
            frame_with_detections = detector.draw_detections(current_frame, detections)
            ret, buffer = cv2.imencode('.jpg', frame_with_detections)
            if not ret:
                continue

            with output_frame_lock:
                output_frame = buffer.tobytes()

        except Exception as e:
            print(f"CRITICAL Error in detection loop: {e}")
            ret, buffer = cv2.imencode('.jpg', current_frame)
            if ret:
                with output_frame_lock:
                    output_frame = buffer.tobytes()
            time.sleep(0.5) 

# --- Flask Routes ---

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():  
    global output_frame, output_frame_lock
    print("Client connected to video stream.")
    
    while True:
        with output_frame_lock:
            if output_frame is None:
                time.sleep(0.1)
                continue
            frame_bytes = output_frame
        
        try:
            # --- THIS IS THE FIX ---
            # Changed \r_n to \r\n
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"Error yielding frame: {e}. Client likely disconnected.")
            break
        
        time.sleep(0.033)
    
    print("Client disconnected from video stream.")


@app.route('/')
def index():
    return render_template('index.html')

@sock.route('/ws')
def ws(client_socket):
    register_client(client_socket)
    try:
        while True:
            data = client_socket.receive(timeout=60)
            if data is None:
                continue
    except Exception as e:
        print(f"WebSocket client error/disconnect: {e}")
    finally:
        unregister_client(client_socket)

# --- Main execution ---

if __name__ == '__main__':
    print("Starting background threads...")
    
    t_camera = threading.Thread(target=read_camera_thread, daemon=True)
    t_detector = threading.Thread(target=run_detection_thread, daemon=True)
    
    t_camera.start()
    print("Waiting for camera to initialize...")
    time.sleep(2.0) # Give the camera 2 seconds to open
    t_detector.start()
    
    print("\n" + "="*50)
    print("Starting production server with Waitress...")
    print(f"Access your app at: http://127.0.0.1:5000/")
    print(f"Or from other devices on your network at: http://[YOUR_IP_ADDRESS]:5000")
    print("="*50 + "\n")
    
    serve(app, host='0.0.0.0', port=5000, threads=10)