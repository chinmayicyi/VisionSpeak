import cv2
from ultralytics import YOLO
import time
import queue
import pyttsx3
import threading
from threading import Lock
import math

class MultiModelDetector:
    def __init__(self, camera_index=0, display_queue=None):
        """
        Optimized detector with queue-based TTS and movement tracking
        """
        print("Loading YOLO models...")

        # Load YOLOv8 models
        self.models = {
            'general': YOLO('yolov8s.pt'),
            'detailed': YOLO('yolov8m.pt'),
        }

        print("Models loaded successfully!")

        # Camera
        self.cap = None
        if camera_index is not None:
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                raise RuntimeError("Could not open camera")

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.running = False
        self.narration_paused = False
        self.display_queue = display_queue

        # Store current frame for web streaming (THREAD-SAFE)
        self.current_frame = None
        self.frame_lock = Lock()

        # TTS QUEUE SYSTEM - Prevents overlap
        self.tts_queue = queue.Queue()
        self.tts_lock = Lock()
        self.speaking = False
        self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.tts_thread.start()
        
        print("TTS queue system initialized")

        # Detection and narration state
        self.last_objects = set()
        self.last_narration_time = 0
        self.min_interval = 2.0  # Minimum seconds between narrations
        self.periodic_interval = 6.0  # Periodic updates

        # Movement tracking
        self.tracked_positions = {}
        self.movement_threshold = 60

        # Detection settings
        self.confidence_threshold = 0.35
        self.min_area = 2000

        # Female voice setup
        self._setup_female_voice()

    def _setup_female_voice(self):
        """Setup female voice (usually index 1 on Windows)"""
        try:
            temp_engine = pyttsx3.init()
            voices = temp_engine.getProperty('voices')
            
            # Try to find female voice
            self.female_voice_id = None
            for i, voice in enumerate(voices):
                # Windows female voice usually contains 'Zira' or is index 1
                if 'female' in voice.name.lower() or 'zira' in voice.name.lower() or i == 1:
                    self.female_voice_id = voice.id
                    print(f"Female voice found: {voice.name}")
                    break
            
            if not self.female_voice_id and len(voices) > 1:
                self.female_voice_id = voices[1].id
                print(f"Using voice at index 1: {voices[1].name}")
            
            temp_engine.stop()
            del temp_engine
        except Exception as e:
            print(f"Voice setup warning: {e}")
            self.female_voice_id = None

    def _tts_worker(self):
        """
        TTS Worker Thread - Processes speech queue sequentially
        Prevents overlapping narration
        """
        while True:
            try:
                text = self.tts_queue.get()
                
                if text is None:  # Shutdown signal
                    break
                
                with self.tts_lock:
                    self.speaking = True
                
                try:
                    # Create fresh engine for each speech (Windows fix)
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 150)
                    engine.setProperty('volume', 0.9)
                    
                    # Set female voice if available
                    if self.female_voice_id:
                        engine.setProperty('voice', self.female_voice_id)
                    
                    print(f"[🔊 Speaking]: {text}")
                    engine.say(text)
                    engine.runAndWait()
                    engine.stop()
                    del engine
                    time.sleep(0.3)
                    
                except Exception as e:
                    print(f"TTS Error: {e}")
                
                finally:
                    with self.tts_lock:
                        self.speaking = False
                    
                    self.tts_queue.task_done()
                    
            except Exception as e:
                print(f"TTS Worker Error: {e}")

    def _speak_async(self, text):
        """Add text to TTS queue (non-blocking)"""
        if not text:
            return
        
        # Add to queue - will be spoken when current speech finishes
        self.tts_queue.put(text)

    def _calculate_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _detect_movement(self, current_detections):
        """Detect and describe object movement"""
        current_time = time.time()
        movements = []
        current_positions = {}

        for det in current_detections:
            name = det['name']
            center = self._calculate_center(det['bbox'])
            current_positions[name] = center

        for name, (curr_x, curr_y) in current_positions.items():
            if name in self.tracked_positions:
                prev_x, prev_y, last_time = self.tracked_positions[name]
                dx = curr_x - prev_x
                dy = curr_y - prev_y
                distance = math.sqrt(dx * dx + dy * dy)

                if distance > self.movement_threshold and (current_time - last_time) > 0.5:
                    # Determine primary direction
                    if abs(dx) > abs(dy):
                        direction = "left" if dx < 0 else "right"
                    else:
                        direction = "up" if dy < 0 else "down"
                    
                    movements.append(f"{name} moving {direction}")
                    print(f"🏃 Movement: {name} moved {distance:.0f}px {direction}")

        # Update tracked positions
        for name, position in current_positions.items():
            self.tracked_positions[name] = (position[0], position[1], current_time)

        # Remove objects that are no longer visible
        for name in list(self.tracked_positions.keys()):
            if name not in current_positions:
                del self.tracked_positions[name]

        return movements

    def _detect_scene_changes(self, current_objects):
        """Detect changes in the scene"""
        current_person_count = sum(1 for obj in current_objects if obj == 'person')
        last_person_count = sum(1 for obj in self.last_objects if obj == 'person')

        current_others = {obj for obj in current_objects if obj != 'person'}
        last_others = {obj for obj in self.last_objects if obj != 'person'}

        changes = []

        # Person changes
        if current_person_count > last_person_count:
            diff = current_person_count - last_person_count
            changes.append("new person detected" if diff == 1 else f"{diff} new persons detected")

        elif current_person_count < last_person_count:
            diff = last_person_count - current_person_count
            changes.append("person left view" if diff == 1 else f"{diff} persons left view")

        # Object changes
        new_objects = current_others - last_others
        removed_objects = last_others - current_others

        for obj in new_objects:
            print(f"➕ New: {obj}")
            changes.append(f"new {obj} detected")

        for obj in removed_objects:
            print(f"➖ Removed: {obj}")
            changes.append(f"{obj} left view")

        return changes

    def _should_narrate(self, detections):
        """Determine if narration should occur"""
        if self.narration_paused:
            return False, None, []

        current_time = time.time()
        time_since_last = current_time - self.last_narration_time
        current_objects = set(det['name'] for det in detections)

        # Check for movement
        movements = self._detect_movement(detections)
        if movements and time_since_last > self.min_interval:
            return True, "movement", movements

        # Check for scene changes
        changes = self._detect_scene_changes(current_objects)
        if changes and time_since_last > self.min_interval:
            return True, "scene_change", changes

        # Periodic update
        if detections and time_since_last > self.periodic_interval:
            return True, "periodic", None

        return False, None, []

    def _create_description(self, detections, special_events=None):
        """
        Create natural language description with aggregation
        Example: "I see 2 persons and a cell phone" instead of "person, person, cell phone"
        """
        if special_events:
            return ", ".join(special_events)

        if not detections:
            return "No objects in view"

        # Count objects
        counts = {}
        for det in detections:
            counts[det['name']] = counts.get(det['name'], 0) + 1

        # Build natural language description
        parts = []
        for obj, count in sorted(counts.items()):
            if obj == "person":
                if count == 1:
                    parts.append("a person")
                else:
                    parts.append(f"{count} persons")
            else:
                if count == 1:
                    parts.append(f"a {obj}")
                else:
                    parts.append(f"{count} {obj}s")

        # Combine parts naturally
        if len(parts) == 1:
            return f"I can see {parts[0]}"
        elif len(parts) == 2:
            return f"I can see {parts[0]} and {parts[1]}"
        else:
            return f"I can see {', '.join(parts[:-1])}, and {parts[-1]}"

    def force_narrate(self):
        """Force immediate narration of current scene"""
        current_objects = list(self.last_objects)
        
        if current_objects:
            # Count objects
            counts = {}
            for obj in current_objects:
                counts[obj] = counts.get(obj, 0) + 1
            
            # Create detection list for proper description
            detections = [{'name': obj} for obj in current_objects]
            description = self._create_description(detections)
            self._speak_async(description)
        else:
            self._speak_async("No objects currently detected")

    def _run_detection(self, frame):
        """Run YOLO detection on a frame"""
        all_detections = []

        for name, model in self.models.items():
            try:
                results = model(frame, conf=self.confidence_threshold, verbose=False)
                for r in results:
                    if r.boxes is None:
                        continue
                    for box in r.boxes:
                        conf = box.conf[0].item()
                        cls = int(box.cls[0].item())
                        obj = r.names[cls]
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        area = (x2-x1)*(y2-y1)

                        if area < self.min_area:
                            continue

                        all_detections.append({
                            'name': obj,
                            'confidence': conf,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)]
                        })
            except:
                continue

        return self._remove_duplicates(all_detections)

    def _remove_duplicates(self, detections):
        """Remove duplicate detections using IOU"""
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        unique = []

        for det in detections:
            duplicate = False
            for existing in unique:
                if self._iou(det['bbox'], existing['bbox']) > 0.6 and det['name'] == existing['name']:
                    duplicate = True
                    break
            if not duplicate:
                unique.append(det)

        return unique

    def _iou(self, b1, b2):
        """Calculate Intersection over Union"""
        x1, y1, x2, y2 = b1
        x3, y3, x4, y4 = b2

        xi1, yi1 = max(x1, x3), max(y1, y3)
        xi2, yi2 = min(x2, x4), min(y2, y4)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0

        inter = (xi2 - xi1) * (yi2 - yi1)
        union = (x2-x1)*(y2-y1) + (x4-x3)*(y4-y3) - inter
        return inter / union if union > 0 else 0

    def _draw_detections(self, frame, detections):
        """Draw bounding boxes on frame"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            name = det['name']
            conf = det['confidence']

            color = (0, 255, 0) if name == "person" else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return frame

    def run(self):
        """Main detection loop - OPTIMIZED"""
        print("============================================================")
        print("🎥 Detection Started - Optimized with Queue-based TTS")
        print("============================================================")

        frame_count = 0

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.03)
                continue

            frame = cv2.flip(frame, 1)
            frame_count += 1

            # OPTIMIZATION: Run detection every 2nd frame only
            if frame_count % 2 == 0:
                detections = self._run_detection(frame)
                annotated_frame = self._draw_detections(frame.copy(), detections)
                
                # Store frame thread-safely
                with self.frame_lock:
                    self.current_frame = annotated_frame

                # Narration logic
                speak, reason, events = self._should_narrate(detections)
                if speak:
                    description = self._create_description(detections, events)
                    self._speak_async(description)
                    self.last_narration_time = time.time()
                    self.last_objects = set(det['name'] for det in detections)
            else:
                # Just draw existing detections on frame (no re-detection)
                with self.frame_lock:
                    if self.current_frame is not None:
                        pass  # Use cached frame
                    else:
                        self.current_frame = frame

            # Display queue (if needed)
            if self.display_queue:
                try:
                    if not self.display_queue.empty():
                        self.display_queue.get_nowait()
                    self.display_queue.put_nowait(self.current_frame)
                except:
                    pass

            time.sleep(0.01)

        if self.cap:
            self.cap.release()
        print("Detection stopped!")