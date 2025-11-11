import torch
import cv2
import pyttsx3
import time
import threading
from ultralytics import YOLO
import numpy as np
from threading import Lock # <-- MEREGED: Added Lock

class MultiModelDetector:
    def __init__(self):
        print("Loading multiple models for comprehensive detection...")
        
        self.models = {
            'general': YOLO('yolov8s.pt'),
            'detailed': YOLO('yolov8m.pt'),
        }
        
        print(f"Loaded {len(self.models)} models")
        for name, model in self.models.items():
            print(f"  {name}: {len(model.names)} classes")
        
        self.all_classes = set()
        for model in self.models.values():
            self.all_classes.update(model.names.values())
        
        print(f"Total unique object types: {len(self.all_classes)}")
        
        # --- MERGED: Upgraded TTS ---
        self.tts_lock = Lock() # Use a lock for thread-safety
        self.speaking = False
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 160)
            self.engine.setProperty('volume', 0.9)
            print("TTS initialized!")
        except Exception as e:
            print(f"TTS Error: {e}")
            self.engine = None
        
        # --- KEPT: GUI Control Flags ---
        self.running = False # This is controlled by app.py
        
        # --- MERGED: New Control Flags ---
        self.narration_paused = False # For spacebar control
        self.show_help = True
        
        # --- MERGED: Upgraded Detection Settings ---
        self.confidence_threshold = 0.4  # Balanced threshold
        self.last_spoken_time = 0
        self.spoken_objects = set()
        self.min_speaking_interval = 3  # Seconds between announcements
        
        # --- KEPT: Camera Setup (app.py controls this) ---
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            # app.py will handle the error message
            return
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
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
                        
                        corrected_name = self.smart_object_correction(class_name, conf, (x2-x1)*(y2-y1))
                        
                        detection = {
                            'name': corrected_name,
                            'confidence': conf,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'model': model_name,
                            'area': (x2-x1) * (y2-y1),
                            'original_name': class_name
                        }
                        all_detections.append(detection)
                        
            except Exception as e:
                print(f"Error with {model_name} model: {e}")
        
        unique_detections = self.remove_duplicates(all_detections)
        return unique_detections
    
    # --- MERGED: Copied smart_object_correction ---
    def smart_object_correction(self, class_name, confidence, area):
        """Apply intelligent corrections for common misdetections"""
        corrections = {
            'donut': {'condition': lambda c, a: a < 5000, 'correction': 'round object', 'reason': 'Small round object'},
            'orange': {'condition': lambda c, a: a < 3000, 'correction': 'round object', 'reason': 'Small round object'},
            'banana': {'condition': lambda c, a: a < 2000, 'correction': 'elongated object', 'reason': 'Small elongated object'},
            'hot dog': {'condition': lambda c, a: a < 1500, 'correction': 'cylindrical object', 'reason': 'Small cylindrical object'},
            'sandwich': {'condition': lambda c, a: a < 8000, 'correction': 'rectangular object', 'reason': 'Rectangular flat object'},
            'pizza': {'condition': lambda c, a: a < 10000, 'correction': 'flat object', 'reason': 'Flat object'},
            'baseball bat': {'condition': lambda c, a: a < 2000, 'correction': 'long object', 'reason': 'Small elongated object'},
            'tennis racket': {'condition': lambda c, a: a < 5000, 'correction': 'handheld object', 'reason': 'Small handheld item'},
            'toothbrush': {'condition': lambda c, a: c < 0.6, 'correction': 'small object', 'reason': 'Low confidence small item'}
        }
        
        if class_name in corrections:
            correction_rule = corrections[class_name]
            if correction_rule['condition'](confidence, area):
                # print(f"🔧 Corrected: {class_name} → {correction_rule['correction']}")
                return correction_rule['correction']
        
        if confidence > 0.7:
            return class_name
        
        if area < 1000:
            food_items = ['donut', 'orange', 'apple', 'banana', 'hot dog']
            if class_name in food_items:
                return 'small object'
        
        return class_name

    # --- MERGED: Copied remove_duplicates ---
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
                if overlap > 0.5 and detection['name'] == existing['name']:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique.append(detection)
        
        return unique

    # --- MERGED: Copied calculate_overlap ---
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

    # --- MERGED: Upgraded speak_text with Lock ---
    def speak_text(self, text):
        """Thread-safe text-to-speech function"""
        if not self.engine:
            return
            
        with self.tts_lock:
            if self.speaking:
                return
            self.speaking = True
            
        try:
            print(f"[🔊 Speaking]: {text}")
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")
        finally:
            with self.tts_lock:
                self.speaking = False

    # --- MERGED: Upgraded create_description with Positional Audio ---
    def create_description(self, detections, frame_width, frame_height):
        """Create natural language description for accessibility"""
        if not detections:
            return "No objects detected in view"
        
        object_counts = {}
        positions = []
        
        for det in detections:
            name = det['name']
            object_counts[name] = object_counts.get(name, 0) + 1
            
            x1, y1, x2, y2 = det['bbox']
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            h_pos = "left" if center_x < frame_width/3 else "right" if center_x > 2*frame_width/3 else "center"
            v_pos = "top" if center_y < frame_height/3 else "bottom" if center_y > 2*frame_height/3 else "middle"
            
            if h_pos == "center" and v_pos == "middle":
                position = "in the center"
            elif h_pos == "center":
                position = f"in the {v_pos}"
            elif v_pos == "middle":
                position = f"on the {h_pos}"
            else:
                position = f"in the {v_pos} {h_pos}"
            
            positions.append(f"{name} {position}")
        
        if len(detections) == 1:
            return f"I see {positions[0]}"
        elif len(detections) <= 3:
            return f"I can see {', '.join(positions[:-1])} and {positions[-1]}"
        else:
            parts = []
            for obj, count in object_counts.items():
                if count == 1:
                    parts.append(f"a {obj}")
                else:
                    parts.append(f"{count} {obj}{'s' if count > 1 else ''}")
            
            if len(parts) <= 2:
                return f"I can see {' and '.join(parts)}"
            else:
                return f"I can see {', '.join(parts[:-1])}, and {parts[-1]}"

    # --- MERGED: Copied draw_detections ---
    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels"""
        colors = {
            'general': (0, 255, 0),
            'detailed': (255, 0, 0),
            'face': (0, 0, 255),
            'pose': (255, 255, 0)
        }
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            name = det['name']
            model = det['model']
            
            color = colors.get(model, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            original_name = det.get('original_name', name)
            if original_name != name:
                label = f"{name} ({conf:.2f}) [was: {original_name}]"
            else:
                label = f"{name} ({conf:.2f})"
                
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return frame

    # --- KEPT & MERGED: The main run loop ---
    # This loop is still controlled by self.running (from app.py)
    # but contains all the upgraded features.
    def run(self):
        if not self.cap or not self.cap.isOpened():
             print("Cannot run: Camera is not open.")
             self.running = False
             return

        print("\n🎥 VisionSpeak Detector Thread Started!")
        print("   Controls are active on the video window:")
        print("     'q' - Quit (also stops GUI)")
        print("     ' ' - Pause/Resume Narration")
        print("     'c' - Cycle Confidence")
        print("     'h' - Toggle Help\n")
        
        frame_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frame_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        
        # This is the main loop, controlled by app.py
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                time.sleep(0.1)
                continue
            
            # MERGED: Add mirror flip
            frame = cv2.flip(frame, 1)
            
            detections = self.detect_with_multiple_models(frame)
            detections = [d for d in detections if d['area'] > 2000]
            
            current_time = time.time()
            current_objects = set([d['name'] for d in detections])
            
            should_speak = False
            if not self.narration_paused and not self.speaking:
                if current_time - self.last_spoken_time > self.min_speaking_interval:
                    if detections:
                        should_speak = True
                elif len(current_objects.symmetric_difference(self.spoken_objects)) > 0:
                    if current_time - self.last_spoken_time > 1:
                        should_speak = True
            
            # --- FIX APPLIED HERE ---
            # Removed 'and detections' to allow "No objects detected" to be spoken
            if should_speak:
                # MERGED: Pass frame dimensions for positional audio
                description = self.create_description(detections, frame_width, frame_height)
                thread = threading.Thread(target=self.speak_text, args=(description,))
                thread.daemon = True
                thread.start()
                
                self.spoken_objects = current_objects.copy()
                self.last_spoken_time = current_time
            
            frame = self.draw_detections(frame, detections)
            
            # --- MERGED: Upgraded Status Overlay ---
            status_color = (0, 255, 0) if not self.speaking else (0, 0, 255)
            speaking_status = "Speaking..." if self.speaking else "Listening"
            narration_status = "PAUSED" if self.narration_paused else "Active"
            
            cv2.putText(frame, f"Objects: {len(detections)} | {speaking_status} | Narration: {narration_status}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            if detections:
                detected_names = ', '.join(list(current_objects)[:5])
                if len(current_objects) > 5:
                    detected_names += "..."
                cv2.putText(frame, f"Detected: {detected_names}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            if self.show_help:
                help_text = [
                    "Controls: Q=Quit, C=Confidence, SPACE=Pause",
                    f"Confidence: {self.confidence_threshold:.2f}"
                ]
                for i, text in enumerate(help_text):
                    cv2.putText(frame, text, (10, int(frame_height) - 40 + i*20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow('VisionSpeak - Object Detection', frame)
            
            # --- MERGED: Upgraded Key Handling ---
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                self.running = False # This flag will stop the loop
            
            elif key == ord('c'):
                thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
                current_idx = thresholds.index(self.confidence_threshold) if self.confidence_threshold in thresholds else 1
                self.confidence_threshold = thresholds[(current_idx + 1) % len(thresholds)]
                print(f"Confidence threshold: {self.confidence_threshold}")
            
            elif key == ord('h'):
                self.show_help = not self.show_help
            
            elif key == ord(' '):
                self.narration_paused = not self.narration_paused
                status = "paused" if self.narration_paused else "resumed"
                print(f"Narration {status}")
        
        # When loop breaks (because self.running = False), run cleanup
        self.cleanup()

    # --- MERGED: Upgraded cleanup ---
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        if self.engine:
            self.engine.stop()
        print("VisionSpeak detector stopped and resources released.")