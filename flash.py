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
        camera_index: webcam index
        display_queue: thread-safe queue to send frames to main thread
        """
        print("Loading YOLO models...")
        
        # Load YOLOv8 models
        self.models = {
            'general': YOLO('yolov8s.pt'),
            'detailed': YOLO('yolov8m.pt'),
        }
        
        print("Models loaded successfully!")

        # Camera
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.running = False
        self.narration_paused = False
        self.display_queue = display_queue

        # TTS engine with threading
        self.tts_lock = Lock()
        self.speaking = False
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)
            self.tts_engine.setProperty('volume', 0.9)
            print("TTS engine initialized!")
        except Exception as e:
            print(f"TTS initialization error: {e}")
            self.tts_engine = None

        # Detection and narration state
        self.last_objects = set()
        self.last_narration_time = 0
        self.narration_cooldown = 3.0  # seconds between narrations
        self.min_change_interval = 2.5  # seconds for scene changes
        self.periodic_interval = 12.0  # periodic updates
        
        # Movement tracking
        self.tracked_positions = {}  # {object_name: (x, y, timestamp)}
        self.movement_threshold = 60  # pixels for significant movement
        self.min_movement_interval = 2.0  # seconds between movement narrations
        
        # Detection settings
        self.confidence_threshold = 0.35
        self.min_area = 2000  # minimum object area in pixels

    def _calculate_center(self, bbox):
        """Calculate center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _detect_movement(self, current_detections):
        """Detect significant object movement"""
        current_time = time.time()
        movements = []
        
        current_positions = {}
        for det in current_detections:
            name = det['name']
            center = self._calculate_center(det['bbox'])
            current_positions[name] = center
        
        # Check for movement
        for name, (curr_x, curr_y) in current_positions.items():
            if name in self.tracked_positions:
                prev_x, prev_y, last_time = self.tracked_positions[name]
                
                # Calculate distance moved
                dx = curr_x - prev_x
                dy = curr_y - prev_y
                distance = math.sqrt(dx*dx + dy*dy)
                
                # Check if movement is significant
                if distance > self.movement_threshold and (current_time - last_time) > 0.5:
                    # Determine direction
                    if abs(dx) > abs(dy):
                        direction = "left" if dx < 0 else "right"
                    else:
                        direction = "up" if dy < 0 else "down"
                    
                    movements.append(f"{name} moving {direction}")
                    print(f"🏃 Movement: {name} moved {distance:.0f}px {direction}")
        
        # Update tracked positions
        for name, position in current_positions.items():
            self.tracked_positions[name] = (position[0], position[1], current_time)
        
        # Clean up old positions
        for name in list(self.tracked_positions.keys()):
            if name not in current_positions:
                del self.tracked_positions[name]
        
        return movements

    def _detect_scene_changes(self, current_objects):
        """Detect changes in scene (new or removed objects)"""
        new_objects = current_objects - self.last_objects
        removed_objects = self.last_objects - current_objects
        
        changes = []
        if new_objects:
            for obj in new_objects:
                changes.append(f"new {obj} detected")
                print(f"➕ New: {obj}")
        
        if removed_objects:
            for obj in removed_objects:
                changes.append(f"{obj} left view")
                print(f"➖ Removed: {obj}")
        
        return changes

    def _speak_async(self, text):
        """Speak text in background thread without blocking"""
        if not self.tts_engine or self.speaking or not text:
            return
        
        with self.tts_lock:
            if self.speaking:
                return
            self.speaking = True
        
        def speak_worker():
            try:
                print(f"[🔊 Speaking]: {text}")
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
                time.sleep(0.2)
            except Exception as e:
                print(f"TTS Error: {e}")
            finally:
                with self.tts_lock:
                    self.speaking = False
        
        thread = threading.Thread(target=speak_worker, daemon=True)
        thread.start()

    def _should_narrate(self, detections):
        """Determine if we should narrate based on multiple triggers"""
        if self.narration_paused or self.speaking:
            return False, None, []
        
        current_time = time.time()
        time_since_last = current_time - self.last_narration_time
        
        current_objects = set(det['name'] for det in detections)
        
        # Priority 1: Movement detection
        movements = self._detect_movement(detections)
        if movements and time_since_last > self.min_movement_interval:
            return True, "movement", movements
        
        # Priority 2: Scene changes (new/removed objects)
        scene_changes = self._detect_scene_changes(current_objects)
        if scene_changes and time_since_last > self.min_change_interval:
            self.last_objects = current_objects.copy()
            return True, "scene_change", scene_changes
        
        # Priority 3: Periodic updates
        if detections and time_since_last > self.periodic_interval:
            self.last_objects = current_objects.copy()
            return True, "periodic", None
        
        return False, None, []

    def _create_description(self, detections, special_events=None):
        """Create natural language description"""
        # Priority for special events (movement/scene changes)
        if special_events:
            if len(special_events) == 1:
                return special_events[0]
            elif len(special_events) == 2:
                return f"{special_events[0]} and {special_events[1]}"
            else:
                return f"{', '.join(special_events[:-1])}, and {special_events[-1]}"
        
        # Regular object description
        if not detections:
            return "No objects in view"
        
        object_counts = {}
        for det in detections:
            name = det['name']
            object_counts[name] = object_counts.get(name, 0) + 1
        
        parts = []
        for obj, count in sorted(object_counts.items()):
            if count == 1:
                parts.append(f"a {obj}")
            else:
                parts.append(f"{count} {obj}s")
        
        if len(parts) == 1:
            return f"I can see {parts[0]}"
        elif len(parts) == 2:
            return f"I can see {parts[0]} and {parts[1]}"
        elif len(parts) <= 4:
            return f"I can see {', '.join(parts[:-1])}, and {parts[-1]}"
        else:
            return f"I can see {len(detections)} objects including {parts[0]}, {parts[1]}, and others"

    def _remove_duplicates(self, detections):
        """Remove duplicate detections from different models"""
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        unique = []
        
        for det in detections:
            is_duplicate = False
            for existing in unique:
                # Calculate IoU
                overlap = self._calculate_iou(det['bbox'], existing['bbox'])
                if overlap > 0.6 and det['name'] == existing['name']:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(det)
        
        return unique

    def _calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union"""
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2
        
        xi1, yi1 = max(x1, x3), max(y1, y3)
        xi2, yi2 = min(x2, x4), min(y2, y4)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x4 - x3) * (y4 - y3)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0

    def _draw_detections(self, frame, detections):
        """Draw bounding boxes and labels on frame"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            name = det['name']
            conf = det['confidence']
            
            # Color based on object type and confidence
            if name == 'person':
                color = (0, 255, 0)  # Green for persons
            elif conf > 0.6:
                color = (255, 0, 255)  # Magenta for high confidence
            elif conf > 0.4:
                color = (0, 255, 255)  # Yellow for medium
            else:
                color = (0, 165, 255)  # Orange for low
            
            # Draw box
            thickness = 3 if conf > 0.6 else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"{name} {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0] + 5, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Draw center point
            center_x, center_y = int((x1+x2)/2), int((y1+y2)/2)
            cv2.circle(frame, (center_x, center_y), 4, color, -1)
        
        return frame

    def run(self):
        """Main detection loop (runs in worker thread)"""
        print("🎥 Detection started!")
        frame_count = 0
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.03)
                continue
            
            frame_count += 1
            frame = cv2.flip(frame, 1)  # Mirror for better UX
            
            # Collect detections from all models
            all_detections = []
            
            for model_name, model in self.models.items():
                try:
                    results = model(frame, conf=self.confidence_threshold, verbose=False)
                    
                    for r in results:
                        boxes = r.boxes
                        if boxes is not None:
                            for i, box in enumerate(boxes):
                                conf = box.conf[0].item()
                                cls_id = int(box.cls[0].item())
                                class_name = r.names[cls_id]
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                
                                area = (x2 - x1) * (y2 - y1)
                                
                                # Filter by area
                                if area < self.min_area:
                                    continue
                                
                                detection = {
                                    'name': class_name,
                                    'confidence': conf,
                                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                    'area': area,
                                    'model': model_name
                                }
                                all_detections.append(detection)
                
                except Exception as e:
                    print(f"Error with {model_name} model: {e}")
                    continue
            
            # Remove duplicates
            unique_detections = self._remove_duplicates(all_detections)
            
            # Draw detections on frame
            annotated_frame = self._draw_detections(frame.copy(), unique_detections)
            
            # Add status overlay
            current_time = time.time()
            time_since_last = current_time - self.last_narration_time
            
            status_color = (0, 0, 255) if self.speaking else (0, 255, 0)
            status_text = "Speaking..." if self.speaking else "Listening"
            narration_text = "PAUSED" if self.narration_paused else "Active"
            
            cv2.putText(annotated_frame, f"Objects: {len(unique_detections)} | {status_text} | Narration: {narration_text}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            cv2.putText(annotated_frame, f"Last spoke: {time_since_last:.1f}s ago",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show detected objects
            if unique_detections:
                obj_names = [d['name'] for d in unique_detections[:5]]
                obj_text = ', '.join(obj_names)
                if len(unique_detections) > 5:
                    obj_text += f" +{len(unique_detections)-5}"
                cv2.putText(annotated_frame, f"Detected: {obj_text}",
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Send annotated frame to main thread
            if self.display_queue:
                try:
                    # Clear old frame if queue is full
                    if not self.display_queue.empty():
                        try:
                            self.display_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.display_queue.put_nowait(annotated_frame)
                except queue.Full:
                    pass
            
            # Check if we should narrate (only every few frames to reduce overhead)
            if frame_count % 3 == 0:
                should_speak, reason, special_events = self._should_narrate(unique_detections)
                
                if should_speak:
                    description = self._create_description(unique_detections, special_events)
                    self._speak_async(description)
                    self.last_narration_time = current_time
                    print(f"🎯 Narrating ({reason}): {description}")
            
            time.sleep(0.01)  # Small delay to prevent CPU overuse
        
        # Cleanup
        print("Stopping detection...")
        self.cap.release()
        print("Detection stopped!")