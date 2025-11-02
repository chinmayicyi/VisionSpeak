import torch
import cv2
import pyttsx3
import time
import threading
from ultralytics import YOLO
import numpy as np
from threading import Lock
import math

class ImprovedMultiModelDetector:
    def __init__(self):
        print("Loading multiple models for comprehensive detection...")
        
        # Load multiple models for different object types
        self.models = {
            'general': YOLO('yolov8s.pt'),      # General objects (80 classes)
            'detailed': YOLO('yolov8m.pt'),     # More detailed detection
        }
        
        print(f"Loaded {len(self.models)} models")
        for name, model in self.models.items():
            print(f"  {name}: {len(model.names)} classes")
        
        # Combined object classes from all models
        self.all_classes = set()
        for model in self.models.values():
            self.all_classes.update(model.names.values())
        
        print(f"Total unique object types: {len(self.all_classes)}")
        
        # TTS setup with threading lock
        self.tts_lock = Lock()
        self.speaking = False
        self.tts_queue = []  # Queue for TTS messages
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            self.engine.setProperty('volume', 0.9)
            voices = self.engine.getProperty('voices')
            if voices:
                self.engine.setProperty('voice', voices[0].id)  # Use first available voice
            print("TTS initialized!")
        except Exception as e:
            print(f"TTS Error: {e}")
            self.engine = None
        
        # IMPROVED NARRATION SETTINGS
        self.confidence_threshold = 0.35  # Balanced threshold
        self.last_spoken_time = 0
        self.last_objects = set()  # Track previous objects for change detection
        self.min_change_interval = 2.5  # Minimum seconds between object change narrations
        self.min_movement_interval = 1.5  # Minimum seconds between movement narrations
        self.periodic_interval = 10  # Narrate every 10 seconds even if no change
        
        # Enhanced tracking state
        self.tracked_objects = {}  # Store previous positions for movement tracking
        self.object_history = {}   # Store detection history for stability
        self.movement_threshold = 50  # Pixels for significant movement
        self.stability_frames = 3  # Frames an object must be present to be considered stable
        
        # Camera setup with better error handling
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            # Try different camera indices
            for i in range(1, 4):
                self.cap = cv2.VideoCapture(i)
                if self.cap.isOpened():
                    print(f"Camera opened on index {i}")
                    break
            else:
                raise RuntimeError("Could not open any camera")
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Initialization complete!")
        
    def detect_with_multiple_models(self, frame):
        """Run detection with all models and combine results"""
        all_detections = []
        
        for model_name, model in self.models.items():
            try:
                results = model(frame, conf=self.confidence_threshold, verbose=False)
                boxes = results[0].boxes
                
                if boxes is not None:
                    for box in boxes:
                        conf = box.conf[0].item()
                        class_id = int(box.cls[0].item())
                        class_name = model.names[class_id]
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        # Apply conservative object corrections
                        corrected_name = self.conservative_object_correction(class_name, conf, (x2-x1)*(y2-y1))
                        
                        detection = {
                            'name': corrected_name,
                            'confidence': conf,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'center': ((x1+x2)/2, (y1+y2)/2),  # Center point for tracking
                            'model': model_name,
                            'area': (x2-x1) * (y2-y1),
                            'original_name': class_name,
                            'id': f"{corrected_name}_{int((x1+x2)/2/100)}_{int((y1+y2)/2/100)}"  # Rough position ID
                        }
                        all_detections.append(detection)
                        
            except Exception as e:
                print(f"Error with {model_name} model: {e}")
                continue
        
        # Remove duplicate detections and apply stability filtering
        unique_detections = self.remove_duplicates(all_detections)
        stable_detections = self.apply_stability_filter(unique_detections)
        return stable_detections
    
    def conservative_object_correction(self, class_name, confidence, area):
        """Apply conservative corrections - only fix obvious mistakes"""
        
        # Only correct very low confidence detections of commonly misdetected items
        if confidence < 0.3:
            corrections = {
                'donut': 'round object' if area < 4000 else class_name,
                'orange': 'round object' if area < 3000 else class_name,
                'banana': 'elongated object' if area < 2000 else class_name,
                'hot dog': 'cylindrical object' if area < 1500 else class_name,
            }
            
            if class_name in corrections:
                corrected = corrections[class_name]
                if corrected != class_name:
                    print(f"🔧 Corrected: {class_name} → {corrected} (low confidence)")
                return corrected
        
        # Trust higher confidence detections, especially for common objects
        return class_name
    
    def apply_stability_filter(self, detections):
        """Only include objects that have been consistently detected"""
        current_time = time.time()
        stable_detections = []
        
        # Update object history
        current_ids = set()
        for det in detections:
            obj_id = det['id']
            current_ids.add(obj_id)
            
            if obj_id not in self.object_history:
                self.object_history[obj_id] = {'count': 1, 'first_seen': current_time, 'last_seen': current_time, 'detection': det}
            else:
                self.object_history[obj_id]['count'] += 1
                self.object_history[obj_id]['last_seen'] = current_time
                self.object_history[obj_id]['detection'] = det  # Update with latest detection
        
        # Remove old objects that haven't been seen recently
        to_remove = []
        for obj_id, history in self.object_history.items():
            if obj_id not in current_ids:
                history['count'] = max(0, history['count'] - 2)  # Decay
                if current_time - history['last_seen'] > 2 or history['count'] <= 0:
                    to_remove.append(obj_id)
        
        for obj_id in to_remove:
            del self.object_history[obj_id]
        
        # Only return objects that have been seen for minimum frames
        for obj_id, history in self.object_history.items():
            if history['count'] >= self.stability_frames:
                stable_detections.append(history['detection'])
        
        return stable_detections
    
    def remove_duplicates(self, detections):
        """Remove overlapping detections from different models"""
        if len(detections) <= 1:
            return detections
        
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        unique = []
        
        for detection in detections:
            is_duplicate = False
            
            for existing in unique:
                overlap = self.calculate_overlap(detection['bbox'], existing['bbox'])
                
                if overlap > 0.6 and detection['name'] == existing['name']:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(detection)
        
        return unique
    
    def calculate_overlap(self, bbox1, bbox2):
        """Calculate IoU between two bounding boxes"""
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
    
    def speak_text(self, text, priority=False):
        """Improved thread-safe text-to-speech function"""
        if not self.engine:
            print(f"[No TTS] {text}")
            return
            
        # If high priority or not currently speaking, speak immediately
        if priority or not self.speaking:
            with self.tts_lock:
                if self.speaking and not priority:
                    return
                self.speaking = True
                
            try:
                print(f"[🔊 Speaking]: {text}")
                self.engine.say(text)
                self.engine.runAndWait()
                time.sleep(0.2)  # Small pause after speaking
            except Exception as e:
                print(f"TTS Error: {e}")
            finally:
                with self.tts_lock:
                    self.speaking = False
    
    def detect_movement(self, current_detections):
        """Enhanced movement detection with better tracking"""
        current_time = time.time()
        movement_descriptions = []
        
        # Create current object positions
        current_positions = {}
        for det in current_detections:
            name = det['name']
            center = det['center']
            current_positions[name] = center
        
        # Compare with previous positions
        for name, (current_x, current_y) in current_positions.items():
            if name in self.tracked_objects:
                prev_x, prev_y, last_time = self.tracked_objects[name]
                
                # Calculate movement distance
                dx = current_x - prev_x
                dy = current_y - prev_y
                distance = math.sqrt(dx*dx + dy*dy)
                
                # Check if movement is significant and enough time has passed
                if distance > self.movement_threshold and (current_time - last_time) > 0.5:
                    # Determine primary direction
                    if abs(dx) > abs(dy):
                        direction = "left" if dx < 0 else "right"
                        movement_descriptions.append(f"{name} moving {direction}")
                    else:
                        direction = "up" if dy < 0 else "down"
                        movement_descriptions.append(f"{name} moving {direction}")
                        
                    print(f"🏃 Movement detected: {name} moved {distance:.0f} pixels {direction}")
        
        # Update tracked positions
        for name, position in current_positions.items():
            self.tracked_objects[name] = (position[0], position[1], current_time)
        
        # Clean up old tracked objects
        current_names = set(current_positions.keys())
        to_remove = []
        for name in self.tracked_objects:
            if name not in current_names:
                to_remove.append(name)
        for name in to_remove:
            del self.tracked_objects[name]
        
        return movement_descriptions
    
    def detect_scene_changes(self, current_detections):
        """Detect significant changes in the scene"""
        current_objects = set(det['name'] for det in current_detections)
        
        # Check for new or removed objects
        new_objects = current_objects - self.last_objects
        removed_objects = self.last_objects - current_objects
        
        changes = []
        if new_objects:
            for obj in new_objects:
                changes.append(f"new {obj} appeared")
                print(f"➕ New object: {obj}")
                
        if removed_objects:
            for obj in removed_objects:
                changes.append(f"{obj} disappeared")
                print(f"➖ Object left: {obj}")
        
        return changes, current_objects
    
    def should_narrate(self, current_detections):
        """Improved decision logic for when to narrate"""
        current_time = time.time()
        time_since_last = current_time - self.last_spoken_time
        
        # Don't interrupt ongoing speech
        if self.speaking:
            return False, "Speaking in progress", []
        
        # Check for movement (highest priority)
        movement_descriptions = self.detect_movement(current_detections)
        if movement_descriptions and time_since_last > self.min_movement_interval:
            return True, "Movement detected", movement_descriptions
        
        # Check for scene changes
        scene_changes, current_objects = self.detect_scene_changes(current_detections)
        if scene_changes and time_since_last > self.min_change_interval:
            self.last_objects = current_objects  # Update tracked objects
            return True, "Scene changed", scene_changes
        
        # Periodic updates if scene is stable
        if current_detections and time_since_last > self.periodic_interval:
            self.last_objects = current_objects
            return True, "Periodic update", []
        
        return False, f"Waiting ({time_since_last:.1f}s)", []
    
    def create_description(self, detections, special_descriptions=None):
        """Create natural language description with priority for special events"""
        if special_descriptions:
            # Priority for movement or scene changes
            if len(special_descriptions) == 1:
                return special_descriptions[0]
            elif len(special_descriptions) == 2:
                return f"{special_descriptions[0]} and {special_descriptions[1]}"
            else:
                return f"{', '.join(special_descriptions[:-1])}, and {special_descriptions[-1]}"
        
        if not detections:
            return "No objects in view"
        
        # Count objects and create description
        object_counts = {}
        for det in detections:
            name = det['name']
            object_counts[name] = object_counts.get(name, 0) + 1
        
        # Create description parts
        parts = []
        for obj, count in sorted(object_counts.items()):
            if count == 1:
                parts.append(f"a {obj}")
            else:
                parts.append(f"{count} {obj}s")
        
        # Construct sentence
        if len(parts) == 1:
            return f"I can see {parts[0]}"
        elif len(parts) == 2:
            return f"I can see {parts[0]} and {parts[1]}"
        elif len(parts) <= 4:
            return f"I can see {', '.join(parts[:-1])}, and {parts[-1]}"
        else:
            # Too many objects, summarize
            return f"I can see {len(detections)} objects including {parts[0]}, {parts[1]}, and others"
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels with improved visuals"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            name = det['name']
            
            # Color based on confidence and object type
            if name == 'person':
                color = (0, 255, 0)  # Green for persons
            elif conf > 0.7:
                color = (255, 0, 255)  # Magenta for high confidence
            elif conf > 0.5:
                color = (0, 255, 255)  # Yellow for medium confidence
            else:
                color = (0, 165, 255)  # Orange for low confidence
            
            # Draw bounding box with thickness based on confidence
            thickness = 3 if conf > 0.6 else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label with background
            original_name = det.get('original_name', name)
            if original_name != name:
                label = f"{name} {conf:.2f} [was: {original_name}]"
            else:
                label = f"{name} {conf:.2f}"
                
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Label background
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                          (x1 + label_size[0] + 5, y1), color, -1)
            
            # Label text
            cv2.putText(frame, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Draw center point for tracking visualization
            center_x, center_y = int(det['center'][0]), int(det['center'][1])
            cv2.circle(frame, (center_x, center_y), 3, color, -1)
        
        return frame
    
    def run(self):
        """Main detection loop with improved narration"""
        print("\n🎥 Enhanced VisionSpeak - Started!")
        print("🔊 Will narrate on scene changes, movement, and periodically")
        print("\nControls:")
        print("  'q' - Quit")
        print("  's' - Force speak current scene")  
        print("  'c' - Change confidence threshold")
        print("  'r' - Reset tracking and force immediate narration")
        print("  SPACE - Pause/Resume narration")
        print("  'h' - Toggle help display\n")
        
        show_help = True
        narration_paused = False
        frame_count = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Failed to capture frame")
                    break
                
                frame = cv2.flip(frame, 1)  # Mirror for better UX
                frame_count += 1
                
                # Detect objects
                detections = self.detect_with_multiple_models(frame)
                
                # Filter by minimum area (more lenient for important objects)
                filtered_detections = []
                for det in detections:
                    if det['name'] == 'person' and det['area'] > 1000:  # Lower threshold for persons
                        filtered_detections.append(det)
                    elif det['name'] in ['book', 'laptop', 'cell phone'] and det['area'] > 1500:  # Important objects
                        filtered_detections.append(det)
                    elif det['area'] > 2500:  # Standard threshold for other objects
                        filtered_detections.append(det)
                
                detections = filtered_detections
                
                # Narration logic (only if not paused)
                if not narration_paused and frame_count % 2 == 0:  # Check every 2 frames
                    should_speak, reason, special_descriptions = self.should_narrate(detections)
                    
                    if should_speak:
                        description = self.create_description(detections, special_descriptions)
                        
                        # Start speaking in background thread
                        thread = threading.Thread(target=self.speak_text, args=(description,))
                        thread.daemon = True
                        thread.start()
                        
                        self.last_spoken_time = time.time()
                        print(f"🎯 Narrating ({reason}): {description}")
                    else:
                        if frame_count % 60 == 0:  # Print reason occasionally
                            print(f"⏸️ Not narrating: {reason}")
                
                # Draw detections
                frame = self.draw_detections(frame, detections)
                
                # Status overlay
                current_time = time.time()
                time_since_last = current_time - self.last_spoken_time
                
                status_color = (0, 0, 255) if self.speaking else (0, 255, 0)
                speaking_status = "Speaking..." if self.speaking else "Listening"
                narration_status = "PAUSED" if narration_paused else "Active"
                
                cv2.putText(frame, f"Objects: {len(detections)} | {speaking_status} | Narration: {narration_status}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                
                # Show last narration time
                cv2.putText(frame, f"Last spoke: {time_since_last:.1f}s ago", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Show detected objects
                if detections:
                    detected_names = [det['name'] for det in detections[:5]]
                    detected_text = ', '.join(detected_names)
                    if len(detections) > 5:
                        detected_text += f" +{len(detections)-5} more"
                    cv2.putText(frame, f"Detected: {detected_text}", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Help text
                if show_help:
                    help_lines = [
                        "Q=Quit | S=Speak | C=Confidence | R=Reset | SPACE=Pause | H=Help",
                        f"Confidence: {self.confidence_threshold:.2f} | Movement: {len(self.tracked_objects)} tracked"
                    ]
                    for i, text in enumerate(help_lines):
                        cv2.putText(frame, text, (10, frame.shape[0] - 40 + i*20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                cv2.imshow('Enhanced VisionSpeak - Object Detection', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('s'):  # Force speak
                    if detections:
                        description = self.create_description(detections)
                        thread = threading.Thread(target=self.speak_text, args=(description, True))  # High priority
                        thread.daemon = True
                        thread.start()
                        self.last_spoken_time = time.time()
                        print(f"🔧 Forced narration: {description}")
                elif key == ord('c'):  # Change confidence
                    thresholds = [0.25, 0.35, 0.45, 0.55, 0.65]
                    try:
                        current_idx = thresholds.index(self.confidence_threshold)
                    except ValueError:
                        current_idx = 1
                    self.confidence_threshold = thresholds[(current_idx + 1) % len(thresholds)]
                    print(f"Confidence threshold: {self.confidence_threshold}")
                elif key == ord('h'):  # Toggle help
                    show_help = not show_help
                elif key == ord(' '):  # Pause/resume narration
                    narration_paused = not narration_paused
                    status = "paused" if narration_paused else "resumed"
                    print(f"Narration {status}")
                elif key == ord('r'):  # Reset tracking
                    self.tracked_objects.clear()
                    self.object_history.clear()
                    self.last_objects = set()
                    self.last_spoken_time = 0
                    print("🔄 Reset tracking - will narrate immediately on next detection")
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Unexpected error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        if self.engine:
            try:
                self.engine.stop()
            except:
                pass
        print("✅ Enhanced VisionSpeak detector stopped.")

if __name__ == "__main__":
    try:
        detector = ImprovedMultiModelDetector()
        detector.run()
    except Exception as e:
        print(f"Failed to start: {e}")
        import traceback
        traceback.print_exc()