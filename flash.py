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
        Windows-compatible detector with TTS fix
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

        # WINDOWS TTS FIX - Reinitialize engine for each speech
        self.tts_lock = Lock()
        self.speaking = False
        print("TTS will be initialized per-speech (Windows fix)")

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
        
        for name, (curr_x, curr_y) in current_positions.items():
            if name in self.tracked_positions:
                prev_x, prev_y, last_time = self.tracked_positions[name]
                
                dx = curr_x - prev_x
                dy = curr_y - prev_y
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance > self.movement_threshold and (current_time - last_time) > 0.5:
                    direction = "left" if abs(dx) > abs(dy) and dx < 0 else \
                               "right" if abs(dx) > abs(dy) and dx > 0 else \
                               "up" if dy < 0 else "down"
                    
                    movements.append(f"{name} moving {direction}")
                    print(f"🏃 Movement: {name} moved {distance:.0f}px {direction}")
        
        for name, position in current_positions.items():
            self.tracked_positions[name] = (position[0], position[1], current_time)
        
        for name in list(self.tracked_positions.keys()):
            if name not in current_positions:
                del self.tracked_positions[name]
        
        return movements

    def _detect_scene_changes(self, current_objects):
        """Detect changes in scene with proper person counting"""
        # Count persons specifically
        current_person_count = sum(1 for obj in current_objects if obj == 'person')
        last_person_count = sum(1 for obj in self.last_objects if obj == 'person')
        
        # Get non-person objects
        current_others = {obj for obj in current_objects if obj != 'person'}
        last_others = {obj for obj in self.last_objects if obj != 'person'}
        
        changes = []
        
        # Handle person count changes
        if current_person_count > last_person_count:
            diff = current_person_count - last_person_count
            if diff == 1:
                changes.append("new person detected")
            else:
                changes.append(f"{diff} new persons detected")
        elif current_person_count < last_person_count:
            diff = last_person_count - current_person_count
            if diff == 1:
                changes.append("person left view")
            else:
                changes.append(f"{diff} persons left view")
        
        # Handle other objects
        new_objects = current_others - last_others
        removed_objects = last_others - current_others
        
        for obj in new_objects:
            changes.append(f"new {obj} detected")
            print(f"➕ New: {obj}")
        
        for obj in removed_objects:
            changes.append(f"{obj} left view")
            print(f"➖ Removed: {obj}")
        
        # Print person count changes
        if current_person_count != last_person_count:
            print(f"👤 Person count changed: {last_person_count} → {current_person_count}")
        
        return changes

    def _speak_async(self, text):
        """WINDOWS FIX: Create new engine for each speech"""
        if not text or self.speaking:
            return
        
        with self.tts_lock:
            if self.speaking:
                return
            self.speaking = True
        
        def speak_worker():
            try:
                print(f"[🔊 Speaking]: {text}")
                
                # WINDOWS FIX: Create fresh engine each time
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)
                engine.setProperty('volume', 0.9)
                
                # Speak
                engine.say(text)
                engine.runAndWait()
                
                # Clean up
                engine.stop()
                del engine
                
                time.sleep(0.3)
                
            except Exception as e:
                print(f"TTS Error: {e}")
            finally:
                with self.tts_lock:
                    self.speaking = False
        
        thread = threading.Thread(target=speak_worker, daemon=True)
        thread.start()

    def _should_narrate(self, detections):
        """Determine if we should narrate"""
        if self.narration_paused:
            return False, None, []
        
        current_time = time.time()
        time_since_last = current_time - self.last_narration_time
        
        current_objects = set(det['name'] for det in detections)
        
        # Priority 1: Movement detection
        movements = self._detect_movement(detections)
        if movements and time_since_last > self.min_interval:
            return True, "movement", movements
        
        # Priority 2: Scene changes
        scene_changes = self._detect_scene_changes(current_objects)
        if scene_changes and time_since_last > self.min_interval:
            return True, "scene_change", scene_changes
        
        # Priority 3: Periodic updates
        if detections and time_since_last > self.periodic_interval:
            return True, "periodic", None
        
        return False, None, []

    def _create_description(self, detections, special_events=None):
        """Create natural language description with proper counting"""
        # Priority for special events (movement/scene changes)
        if special_events:
            # Count persons in special events
            person_events = [e for e in special_events if 'person' in e]
            other_events = [e for e in special_events if 'person' not in e]
            
            # Count how many "new person" events
            person_count = sum(1 for e in person_events if 'new person' in e)
            
            # Create better descriptions for multiple persons
            if person_count > 1:
                description_parts = [f"{person_count} persons detected"]
                description_parts.extend(other_events)
            else:
                description_parts = person_events + other_events
            
            if len(description_parts) == 1:
                return description_parts[0]
            elif len(description_parts) == 2:
                return f"{description_parts[0]} and {description_parts[1]}"
            else:
                return f"{', '.join(description_parts[:-1])}, and {description_parts[-1]}"
        
        # Regular object description with proper counting
        if not detections:
            return "No objects in view"
        
        object_counts = {}
        for det in detections:
            name = det['name']
            object_counts[name] = object_counts.get(name, 0) + 1
        
        # Create description parts with proper pluralization
        parts = []
        for obj, count in sorted(object_counts.items()):
            if obj == 'person':
                # Special handling for persons
                if count == 1:
                    parts.append("a person")
                else:
                    parts.append(f"{count} persons")
            else:
                # Regular objects
                if count == 1:
                    parts.append(f"a {obj}")
                else:
                    # Proper pluralization
                    parts.append(f"{count} {obj}s")
        
        # Construct natural sentence
        if len(parts) == 1:
            return f"I can see {parts[0]}"
        elif len(parts) == 2:
            return f"I can see {parts[0]} and {parts[1]}"
        elif len(parts) <= 4:
            return f"I can see {', '.join(parts[:-1])}, and {parts[-1]}"
        else:
            # Too many object types, summarize
            total = len(detections)
            return f"I can see {total} objects including {parts[0]}, {parts[1]}, and others"

    def _remove_duplicates(self, detections):
        """Remove duplicate detections"""
        if len(detections) <= 1:
            return detections
        
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        unique = []
        
        for det in detections:
            is_duplicate = False
            for existing in unique:
                overlap = self._calculate_iou(det['bbox'], existing['bbox'])
                if overlap > 0.6 and det['name'] == existing['name']:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(det)
        
        return unique

    def _calculate_iou(self, bbox1, bbox2):
        """Calculate IoU"""
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
        """Draw bounding boxes and labels"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            name = det['name']
            conf = det['confidence']
            
            if name == 'person':
                color = (0, 255, 0)
            elif conf > 0.6:
                color = (255, 0, 255)
            elif conf > 0.4:
                color = (0, 255, 255)
            else:
                color = (0, 165, 255)
            
            thickness = 3 if conf > 0.6 else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            label = f"{name} {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0] + 5, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            center_x, center_y = int((x1+x2)/2), int((y1+y2)/2)
            cv2.circle(frame, (center_x, center_y), 4, color, -1)
        
        return frame

    def run(self):
        """Main detection loop"""
        print("="*60)
        print("🎥 Detection Started - Windows TTS Fix Applied")
        print("="*60)
        print("Will narrate:")
        print("  ✅ Scene changes (every 2s)")
        print("  ✅ Movement detected")
        print("  ✅ Periodic updates (every 6s)")
        print("="*60)
        
        frame_count = 0
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.03)
                continue
            
            frame_count += 1
            frame = cv2.flip(frame, 1)
            
            # Collect detections
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
            
            # Draw detections
            annotated_frame = self._draw_detections(frame.copy(), unique_detections)
            
            # Calculate object counts for display
            object_counts = {}
            total_objects = len(unique_detections)
            
            for det in unique_detections:
                name = det['name']
                object_counts[name] = object_counts.get(name, 0) + 1
            
            # Add status overlay - LARGER AND MORE PROMINENT
            current_time = time.time()
            time_since_last = current_time - self.last_narration_time
            
            # TOP BANNER - Total Objects Count
            banner_height = 80
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (0, 0), (annotated_frame.shape[1], banner_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
            
            # LARGE total count
            total_text = f"TOTAL OBJECTS: {total_objects}"
            cv2.putText(annotated_frame, total_text, (20, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            
            # Status indicator
            status_color = (0, 0, 255) if self.speaking else (0, 255, 0)
            status_text = "SPEAKING..." if self.speaking else "LISTENING"
            narration_text = "PAUSED" if self.narration_paused else "ACTIVE"
            
            cv2.putText(annotated_frame, f"Status: {status_text}", 
                       (annotated_frame.shape[1] - 300, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
            cv2.putText(annotated_frame, f"Narration: {narration_text}", 
                       (annotated_frame.shape[1] - 300, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # DETAILED BREAKDOWN - Below banner
            y_offset = banner_height + 30
            
            if object_counts:
                # Show breakdown of each object type
                breakdown_text = "Detected: "
                parts = []
                for obj, count in sorted(object_counts.items()):
                    if obj == 'person':
                        parts.append(f"{count} {'person' if count == 1 else 'persons'}")
                    else:
                        parts.append(f"{count} {obj}{'s' if count > 1 else ''}")
                
                breakdown_text += ", ".join(parts)
                
                # Draw breakdown with background
                text_size = cv2.getTextSize(breakdown_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(annotated_frame, (10, y_offset - 25), 
                            (20 + text_size[0], y_offset + 5), (0, 0, 0), -1)
                cv2.rectangle(annotated_frame, (10, y_offset - 25), 
                            (20 + text_size[0], y_offset + 5), (0, 255, 255), 2)
                
                cv2.putText(annotated_frame, breakdown_text, (15, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Last spoke time
            cv2.putText(annotated_frame, f"Last spoke: {time_since_last:.1f}s ago",
                       (15, y_offset + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Send frame
            if self.display_queue:
                try:
                    if not self.display_queue.empty():
                        try:
                            self.display_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.display_queue.put_nowait(annotated_frame)
                except queue.Full:
                    pass
            
            # Check if we should narrate
            if frame_count % 3 == 0:
                current_objects = set(det['name'] for det in unique_detections)
                
                # Debug info every second
                if frame_count % 30 == 0:
                    print(f"🔍 Current: {current_objects} | Last: {self.last_objects} | Time: {time_since_last:.1f}s | Speaking: {self.speaking}")
                
                should_speak, reason, special_events = self._should_narrate(unique_detections)
                
                if should_speak:
                    description = self._create_description(unique_detections, special_events)
                    self._speak_async(description)
                    self.last_narration_time = current_time
                    
                    # Update last_objects AFTER narrating
                    self.last_objects = current_objects.copy()
                    
                    print(f"🎯 Narrating ({reason}): {description}")
            
            time.sleep(0.01)
        
        # Cleanup
        print("Stopping detection...")
        self.cap.release()
        print("Detection stopped!")