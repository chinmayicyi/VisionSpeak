import torch
import cv2
import pyttsx3
import time
import threading
from ultralytics import YOLO
import numpy as np

class MultiModelDetector:
    def _init_(self):
        print("Loading multiple models for comprehensive detection...")
        
        # Load multiple models for different object types
        self.models = {
            'general': YOLO('yolov8s.pt'),      # General objects (80 classes)
            'detailed': YOLO('yolov8m.pt'),     # More detailed detection
        }
        
        # You can add more specialized models:
        # 'face': YOLO('yolov8n-face.pt'),    # Face detection
        # 'pose': YOLO('yolov8n-pose.pt'),    # Human pose
        
        print(f"Loaded {len(self.models)} models")
        for name, model in self.models.items():
            print(f"  {name}: {len(model.names)} classes")
        
        # Combined object classes from all models
        self.all_classes = set()
        for model in self.models.values():
            self.all_classes.update(model.names.values())
        
        print(f"Total unique object types: {len(self.all_classes)}")
        
        # TTS setup
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            print("TTS initialized!")
        except Exception as e:
            print(f"TTS Error: {e}")
            self.engine = None
        
        # New: Control flag for the run loop (from patch)
        self.running = False
        
        # Detection settings
        self.confidence_threshold = 0.35  # Lower for more detections
        self.speaking = False
        self.last_spoken_time = 0
        self.spoken_objects = set()
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
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
                        
                        # Apply smart object corrections
                        corrected_name = self.smart_object_correction(class_name, conf, (x2-x1)*(y2-y1))
                        
                        detection = {
                            'name': corrected_name,
                            'confidence': conf,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'model': model_name,
                            'area': (x2-x1) * (y2-y1),
                            'original_name': class_name  # Keep original for debugging
                        }
                        all_detections.append(detection)
                        
            except Exception as e:
                print(f"Error with {model_name} model: {e}")
        
        # Remove duplicate detections (same object detected by multiple models)
        unique_detections = self.remove_duplicates(all_detections)
        return unique_detections
    
    def smart_object_correction(self, class_name, confidence, area):
        """Apply intelligent corrections for common misdetections"""
        
        # Common misdetections and their likely corrections
        corrections = {
            # Food items misdetected as tech items (very common!)
            'donut': {
                'condition': lambda c, a: a < 5000,  # Small area
                'correction': 'computer mouse',
                'reason': 'Small round object likely mouse, not donut'
            },
            'orange': {
                'condition': lambda c, a: a < 3000,
                'correction': 'small round object',
                'reason': 'Small orange object'
            },
            'banana': {
                'condition': lambda c, a: a < 2000,
                'correction': 'pen or pencil',
                'reason': 'Small elongated object'
            },
            'hot dog': {
                'condition': lambda c, a: a < 1500,
                'correction': 'pen',
                'reason': 'Small cylindrical object'
            },
            'sandwich': {
                'condition': lambda c, a: a < 8000,
                'correction': 'book or notebook',
                'reason': 'Rectangular flat object'
            },
            'pizza': {
                'condition': lambda c, a: a < 10000,
                'correction': 'flat object',
                'reason': 'Flat circular/triangular object'
            },
            
            # Sports items misdetected
            'baseball bat': {
                'condition': lambda c, a: a < 2000,
                'correction': 'pen or stick',
                'reason': 'Small elongated object'
            },
            'tennis racket': {
                'condition': lambda c, a: a < 5000,
                'correction': 'handheld object',
                'reason': 'Small handheld item'
            },
            
            # Low confidence corrections
            'toothbrush': {
                'condition': lambda c, a: c < 0.6,
                'correction': 'small object',
                'reason': 'Low confidence small item'
            }
        }
        
        # Apply corrections
        if class_name in corrections:
            correction_rule = corrections[class_name]
            if correction_rule['condition'](confidence, area):
                print(f"🔧 Corrected: {class_name} → {correction_rule['correction']} ({correction_rule['reason']})")
                return correction_rule['correction']
        
        # High confidence detections - trust them
        if confidence > 0.8:
            return class_name
        
        # Size-based generic corrections for very small objects
        if area < 1000:
            food_items = ['donut', 'orange', 'apple', 'banana', 'hot dog']
            if class_name in food_items:
                return 'small object'
        
        return class_name
    
    def remove_duplicates(self, detections):
        """Remove overlapping detections from different models"""
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence (highest first)
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        unique = []
        
        for detection in detections:
            is_duplicate = False
            
            for existing in unique:
                # Calculate overlap
                overlap = self.calculate_overlap(detection['bbox'], existing['bbox'])
                
                # If high overlap and same class, it's a duplicate
                if overlap > 0.5 and detection['name'] == existing['name']:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(detection)
        
        return unique
    
    def calculate_overlap(self, bbox1, bbox2):
        """Calculate IoU (Intersection over Union) between two bounding boxes"""
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2
        
        # Calculate intersection
        xi1, yi1 = max(x1, x3), max(y1, y3)
        xi2, yi2 = min(x2, x4), min(y2, y4)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        
        # Calculate union
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x4 - x3) * (y4 - y3)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def speak_text(self, text):
        """Speak text in separate thread"""
        if self.engine and not self.speaking:
            print("self.speaking" + str(self.speaking))
            self.speaking = True
            try:
                print(f"[🔊 Speaking]: {text}")
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print(f"TTS Error: {e}")
            finally:
                self.speaking = False
    
    def create_description(self, detections):
        """Create natural language description"""
        if not detections:
            return "No objects detected"
        
        # Count each object type
        object_counts = {}
        for det in detections:
            name = det['name']
            object_counts[name] = object_counts.get(name, 0) + 1
        
        # Create natural description
        parts = []
        for obj, count in object_counts.items():
            if count == 1:
                parts.append(f"a {obj}")
            else:
                parts.append(f"{count} {obj}s")
        
        # Construct sentence
        if len(parts) == 1:
            return f"I can see {parts[0]}"
        elif len(parts) == 2:
            return f"I can see {parts[0]} and {parts[1]}"
        else:
            return f"I can see {', '.join(parts[:-1])}, and {parts[-1]}"
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels"""
        colors = {
            'general': (0, 255, 0),    # Green
            'detailed': (255, 0, 0),   # Blue
            'face': (0, 0, 255),       # Red
            'pose': (255, 255, 0)      # Cyan
        }
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            name = det['name']
            model = det['model']
            
            color = colors.get(model, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            original_name = det.get('original_name', name)
            if original_name != name:
                label = f"{name} {conf:.2f} [was: {original_name}] [{model}]"
            else:
                label = f"{name} {conf:.2f} [{model}]"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Label background
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Label text
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return frame
    
    def run(self):
        """Main detection loop - now controlled by self.running"""
        # This check is from the patch
        if not self.running:
            print("Detector is not set to run. Call start_detector() first.")
            return

        print("Multi-Model Object Detection Started!")
        print("Press 'q' to quit, 's' to force speak, 'c' to change confidence")
        
        show_model_info = True
        
        # Loop is now controlled by self.running (from patch)
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Detect objects with all models
            detections = self.detect_with_multiple_models(frame)
            
            # Filter by minimum area
            detections = [d for d in detections if d['area'] > 1000]
            
            # Check if we should announce
            now = time.time()
            current_objects = set([d['name'] for d in detections])
            
            should_speak = False
            if now - self.last_spoken_time > 4:
                should_speak = True
            elif current_objects != self.spoken_objects and current_objects:
                should_speak = True
            
            if should_speak and not self.speaking and detections:
                description = self.create_description(detections)
                thread = threading.Thread(target=self.speak_text, args=(description,))
                thread.daemon = True
                thread.start()
                
                self.spoken_objects = current_objects
                self.last_spoken_time = now
            
            # Draw detections
            frame = self.draw_detections(frame, detections)
            
            # Status overlay
            status = f"Objects: {len(detections)} | Models: {len(self.models)} | Confidence: {self.confidence_threshold:.2f}"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if detections:
                detected_names = ', '.join(current_objects)
                cv2.putText(frame, f"Detected: {detected_names}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Show available classes (first few seconds)
            if show_model_info and time.time() < 10:
                sample_classes = list(self.all_classes)[:8]  # Show first 8
                classes_text = f"Available: {', '.join(sample_classes)}..."
                cv2.putText(frame, classes_text, (10, frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow('Multi-Model VisionSpeak', frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False  # Set flag to stop loop (from patch)
                break
            elif key == ord('s'):  # Force speak
                if detections and not self.speaking:
                    description = self.create_description(detections)
                    thread = threading.Thread(target=self.speak_text, args=(description,))
                    thread.daemon = True
                    thread.start()
            elif key == ord('c'):  # Change confidence
                self.confidence_threshold = 0.2 if self.confidence_threshold > 0.3 else 0.4
                print(f"Confidence threshold: {self.confidence_threshold}")
            elif key == ord('h'):  # Hide/show model info
                show_model_info = not show_model_info
            
            # Add a small sleep to yield control (from patch)
            time.sleep(0.01)
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Multi-model detector stopped.")

# The if _name_ == "_main_": block has been removed as requested by the patch
# This file is now ready to be imported by app.py