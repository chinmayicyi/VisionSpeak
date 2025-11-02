import torch
import cv2
import pyttsx3
import time
import threading
from ultralytics import YOLO
import numpy as np

class MultiModelDetector:
    def __init__(self):
        print("Loading multiple models for comprehensive detection...")
        
        # Load multiple models for different object types
        # Using yolov8n.pt (nano) for general and yolov8s.pt (small) for detailed
        # These are generally good for real-time performance on average hardware.
        # You can revert to yolov8s.pt and yolov8m.pt if your hardware is powerful.
        self.models = {
            'general': YOLO('yolov8n.pt'),    # General objects (80 classes)
            'detailed': YOLO('yolov8s.pt'),   # More detailed detection
            # 'face': YOLO('yolov8n-face.pt'), # Uncomment if you have this model for face detection
            # 'pose': YOLO('yolov8n-pose.pt'), # Uncomment if you have this model for human pose
        }
        
        print(f"Loaded {len(self.models)} models")
        for name, model in self.models.items():
            print(f"  - {name}: {len(model.names)} classes")
        
        # Combined set of all unique object classes from all loaded models
        self.all_classes = set()
        for model in self.models.values():
            self.all_classes.update(model.names.values())
        
        print(f"Total unique object types across all models: {len(self.all_classes)}")
        
        # TTS setup
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150) # Speech speed
            # Optional: Set voice (uncomment and test on your OS)
            # voices = self.engine.getProperty('voices')
            # if voices:
            #     # Try different indices (0, 1, etc.) to find a preferred voice
            #     # Note: Voice IDs vary by OS. You might need to print voices to see options.
            #     # Example for Windows: self.engine.setProperty('voice', voices[0].id) 
            #     # Example for macOS: self.engine.setProperty('voice', 'com.apple.speech.synthesis.voice.Alex')
            #     self.engine.setProperty('voice', voices[0].id) 
            #     print(f"Using voice: {self.engine.getProperty('voice')}")
            # else:
            #     print("No specific voices found, using default.")

            print("TTS initialized successfully!")
        except Exception as e:
            print(f"TTS Error during initialization: {e}. Narration will be disabled.")
            self.engine = None
        
        # Detection settings
        self.confidence_threshold = 0.45  # Default confidence for detections
        self.last_spoken_time = 0         # Timestamp of last narration
        self.spoken_objects = set()       # Objects currently being narrated or recently narrated
        self.speaking_lock = threading.Lock() # Lock to ensure only one speech thread runs at a time
        
        # Priority map for narration: objects with higher values are prioritized
        self.priority_map = {
            'person': 10,
            'dog': 9, 'cat': 9,           # Pets
            'car': 8, 'truck': 8, 'bus': 8, # Vehicles
            'door': 7, 'window': 7,       # Important environmental features
            'chair': 6, 'table': 6,       # Furniture
            'backpack': 5, 'handbag': 5,  # Personal items
            'cell phone': 4, 'remote': 4, # Common handheld electronics
            # Add more specific priorities as needed
        }
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam. Please check connection/permissions. Make sure camera is not in use by other apps.")
            exit()
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print("Webcam initialized!")
        
    def detect_with_multiple_models(self, frame):
        """Runs detection with all loaded models and combines their results.
           Applies smart corrections and removes duplicate bounding boxes."""
        all_detections = []
        
        for model_name, model in self.models.items():
            try:
                # Run inference for the current model
                results = model(frame, conf=self.confidence_threshold, verbose=False)
                boxes = results[0].boxes
                
                if boxes is not None:
                    for box in boxes:
                        conf = box.conf[0].item()
                        class_id = int(box.cls[0].item())
                        class_name = model.names[class_id]
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        # Calculate area for smart correction and filtering
                        area = (x2 - x1) * (y2 - y1)
                        
                        # Apply intelligent object corrections based on heuristics
                        corrected_name = self.smart_object_correction(class_name, conf, area)
                        
                        detection = {
                            'name': corrected_name,
                            'confidence': conf,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'model': model_name,
                            'area': area,
                            'original_name': class_name  # Keep original for debugging display
                        }
                        all_detections.append(detection)
                        
            except Exception as e:
                print(f"Error with {model_name} model during detection: {e}")
        
        # Remove overlapping detections (e.g., same object detected by multiple models)
        unique_detections = self.remove_duplicates(all_detections)
        return unique_detections
    
    def smart_object_correction(self, class_name, confidence, area):
        """Applies intelligent corrections for common misdetections based on confidence and size."""
        
        # Define correction rules:
        # 'condition': a lambda function taking confidence (c) and area (a)
        # 'correction': the new, corrected name for the object
        # 'reason': explanation for the correction (for debugging)
        corrections = {
            # Food items often misdetected as small tech/household items
            'donut': {
                'condition': lambda c, a: a < 6000 and c < 0.7, 
                'correction': 'computer mouse',
                'reason': 'Small, low-confidence donut often a mouse'
            },
            'orange': {
                'condition': lambda c, a: a < 4000 and c < 0.65,
                'correction': 'small round object',
                'reason': 'Small, low-confidence orange'
            },
            'banana': {
                'condition': lambda c, a: a < 3000 and c < 0.6,
                'correction': 'pen or pencil',
                'reason': 'Small, low-confidence banana often a pen'
            },
            'hot dog': {
                'condition': lambda c, a: a < 2000 and c < 0.6,
                'correction': 'pen',
                'reason': 'Small, low-confidence hot dog often a pen'
            },
            'sandwich': {
                'condition': lambda c, a: a < 10000 and c < 0.6,
                'correction': 'book or notebook',
                'reason': 'Rectangular, low-confidence sandwich'
            },
            'pizza': {
                'condition': lambda c, a: a < 12000 and c < 0.6,
                'correction': 'flat object',
                'reason': 'Flat, low-confidence pizza'
            },
            
            # Sports items misdetected
            'baseball bat': {
                'condition': lambda c, a: a < 3000 and c < 0.6,
                'correction': 'stick',
                'reason': 'Small, low-confidence baseball bat'
            },
            'tennis racket': {
                'condition': lambda c, a: a < 7000 and c < 0.6,
                'correction': 'handheld object',
                'reason': 'Small, low-confidence tennis racket'
            },
            
            # Generic low confidence/small object corrections
            'toothbrush': {
                'condition': lambda c, a: c < 0.5 and a < 1500,
                'correction': 'small object',
                'reason': 'Very small or low-confidence toothbrush'
            },
            'cup': {
                'condition': lambda c, a: c < 0.5 and a < 2000,
                'correction': 'small container',
                'reason': 'Very small or low-confidence cup'
            }
        }
        
        # Apply corrections if rules are met
        if class_name in corrections:
            rule = corrections[class_name]
            if rule['condition'](confidence, area):
                # Uncomment the line below to see correction messages in console
                # print(f"🔧 Corrected: '{class_name}' ({confidence:.2f}, Area:{area}) -> '{rule['correction']}' ({rule['reason']})")
                return rule['correction']
        
        # Trust high confidence detections without correction
        if confidence > 0.85:
            return class_name
        
        # Generic correction for very small, ambiguous objects with lower confidence
        if area < 800 and confidence < 0.6: 
            ambiguous_small_objects = ['cell phone', 'remote', 'mouse', 'keyboard', 'book', 'bottle', 'cup']
            if class_name in ambiguous_small_objects:
                # print(f"🔧 Corrected: '{class_name}' ({confidence:.2f}, Area:{area}) -> 'tiny object' (generic small object)")
                return 'tiny object'

        return class_name
    
    def remove_duplicates(self, detections):
        """Removes overlapping bounding boxes, keeping the one with the highest confidence."""
        if len(detections) <= 1:
            return detections
        
        # Sort detections by confidence in descending order to prioritize more confident boxes
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        unique_detections = []
        
        for det1 in detections:
            is_duplicate = False
            for det2 in unique_detections:
                # Calculate IoU (Intersection over Union) between current and already unique detections
                overlap = self.calculate_overlap(det1['bbox'], det2['bbox'])
                
                # If high overlap and they represent the same *corrected* class, consider it a duplicate.
                # Adjust IoU threshold (0.6) based on how aggressively you want to merge.
                if overlap > 0.6 and det1['name'] == det2['name']:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_detections.append(det1)
        
        return unique_detections
    
    def calculate_overlap(self, bbox1, bbox2):
        """Calculates IoU (Intersection over Union) between two bounding boxes."""
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2
        
        # Calculate intersection coordinates
        xi1, yi1 = max(x1, x3), max(y1, y3)
        xi2, yi2 = min(x2, x4), min(y2, y4)
        
        # Calculate intersection area
        intersection_width = max(0, xi2 - xi1)
        intersection_height = max(0, yi2 - yi1)
        intersection_area = intersection_width * intersection_height
        
        # Calculate union area
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x4 - x3) * (y4 - y3)
        union_area = area1 + area2 - intersection_area
        
        # Return IoU (Intersection / Union); handle division by zero
        return intersection_area / union_area if union_area > 0 else 0
    
    def speak_text(self, text):
        """Speaks text in a separate thread. Uses a lock to prevent speech overlaps."""
        if self.engine:
            with self.speaking_lock: # Acquire lock to ensure thread-safe access to engine
                if not self.engine.isBusy(): # Check if the TTS engine is currently speaking
                    print(f"[🔊 Speaking]: {text}")
                    try:
                        self.engine.say(text)
                        self.engine.runAndWait() # This blocks the thread until speech is done
                        # Small pause to allow engine state to clear, prevents rapid re-triggering
                        time.sleep(0.1) 
                    except Exception as e:
                        print(f"TTS runtime error: {e}")
                else:
                    print(f"[🗣️ Busy, skipping]: {text}")
    
    def get_spatial_descriptor(self, bbox, frame_width):
        """Determines if an object is on the left, center, or right of the frame."""
        x1, _, x2, _ = bbox
        center_x = (x1 + x2) / 2
        
        # Divide the frame into three equal sections
        if center_x < frame_width / 3:
            return "on your left"
        elif center_x > 2 * frame_width / 3:
            return "on your right"
        else:
            return "in the center"

    def create_description(self, detections, frame_width):
        """Creates a natural language description, prioritizing important objects and adding spatial info."""
        if not detections:
            return "No objects detected"
        
        # Group and count objects, also store their spatial descriptor and priority
        object_info = {}
        for det in detections:
            name = det['name']
            spatial = self.get_spatial_descriptor(det['bbox'], frame_width)
            priority = self.priority_map.get(name, 1) # Default priority 1 if not in map
            
            if name not in object_info:
                object_info[name] = {'count': 0, 'spatial': {}, 'priority': priority}
            
            object_info[name]['count'] += 1
            # Track spatial counts for each object type
            object_info[name]['spatial'][spatial] = object_info[name]['spatial'].get(spatial, 0) + 1
        
        # Sort objects by their defined priority (descending: highest priority first)
        sorted_objects = sorted(object_info.items(), key=lambda item: item[1]['priority'], reverse=True)
        
        parts = []
        for obj_name, info in sorted_objects:
            count = info['count']
            
            # Determine the most prominent spatial descriptor(s) for the object type
            spatial_descriptors = []
            if len(info['spatial']) == 1:
                # If all instances are in one spatial area, just use that area
                spatial_descriptors.append(list(info['spatial'].keys())[0])
            else:
                # If instances are spread out, mention counts for each spatial area
                for sp_desc, sp_count in info['spatial'].items():
                    if sp_count == count: # If all instances are in this one spot
                        spatial_descriptors.append(sp_desc)
                    elif sp_count > 0: # Otherwise, specify count for that area
                        spatial_descriptors.append(f"{sp_count} {sp_desc}")

            spatial_phrase = ""
            if spatial_descriptors:
                if len(spatial_descriptors) == 1:
                    spatial_phrase = f" {spatial_descriptors[0]}"
                else:
                    spatial_phrase = f" ({'; '.join(spatial_descriptors)})" # E.g., "(2 on your left; 1 in the center)"

            if count == 1:
                parts.append(f"a {obj_name}{spatial_phrase}")
            else:
                parts.append(f"{count} {obj_name}s{spatial_phrase}") # Pluralize objects
        
        # Construct the final descriptive sentence
        if len(parts) == 1:
            return f"I can see {parts[0]}."
        elif len(parts) == 2:
            return f"I can see {parts[0]} and {parts[1]}."
        else:
            return f"I can see {', '.join(parts[:-1])}, and {parts[-1]}."
    
    def draw_detections(self, frame, detections):
        """Draws bounding boxes, confidence, model source, and original name (if corrected) on the frame."""
        # Define colors for different models for visual distinction
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
            model_source = det['model']
            original_name = det.get('original_name', name) # Get original name, defaults to corrected if not present
            
            color = colors.get(model_source, (255, 255, 255)) # Default to white if model color not defined
            
            # Draw bounding box rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare the label text for display
            label = f"{name} ({conf:.2f})"
            if original_name != name: # Indicate if the object name was corrected
                label += f" [was:{original_name}]"
            label += f" [{model_source}]" # Show which model detected it
            
            # Get text size to draw a background rectangle for better readability
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            
            # Draw label background rectangle just above the bounding box
            cv2.rectangle(frame, (x1, y1 - text_height - 10), 
                          (x1 + text_width + 10, y1), color, -1)
            
            # Draw the label text on the background
            cv2.putText(frame, label, (x1 + 5, y1 - 5), 
                        font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
        
        return frame
    
    def run(self):
        """Main real-time detection and narration loop."""
        print("\n--- VisionSpeak Real-time Object Detection Started ---")
        print("Controls:")
        print("  'q'     : Quit the application.")
        print("  's'     : Force immediate narration of detected objects.")
        print("  'c'     : Toggle confidence threshold (0.45 <-> 0.25).")
        print("  'h'     : Toggle display of available classes and instructions.")
        
        show_info_overlay = False # Initially hide available classes
        self.start_time = time.time() # Used to display info for initial seconds
        
        # --- ADDED DEBUGGING PRINTS ---
        print("\nDEBUG: Entering main loop...")
        
        while True:
            # --- DEBUG: BEFORE CAP.READ ---
            # print("DEBUG: Attempting to read frame...") # Uncomment for very granular debug
            ret, frame = self.cap.read()
            # --- DEBUG: AFTER CAP.READ ---
            # print(f"DEBUG: Cap.read() returned ret={ret}") # Uncomment for very granular debug

            if not ret:
                print("Failed to grab frame. Exiting application. Check webcam connection.")
                break
            
            frame_width = frame.shape[1] # Get frame width for spatial awareness calculation
            
            # Detect objects using all models and get unique, corrected detections
            detections = self.detect_with_multiple_models(frame)
            
            # Filter out very small detections that might be noise or irrelevant
            detections = [d for d in detections if d['area'] > 1500] # Minimum area threshold for relevance
            
            # --- Narration Logic ---
            now = time.time()
            current_objects_set = set([d['name'] for d in detections]) # Set of names for comparison
            
            should_speak = False
            # Condition 1: Speak periodically (every 5 seconds) if objects are present
            if now - self.last_spoken_time > 5: 
                should_speak = True
            # Condition 2: Speak immediately if new distinct objects appear in the scene
            elif current_objects_set and self.spoken_objects != current_objects_set:
                should_speak = True
            
            # --- DEBUG: NARATION CONDITIONS ---
            # Uncomment for detailed narration trigger info
            # print(f"DEBUG: Should speak: {should_speak}, Detections: {len(detections)}, Engine: {self.engine is not None}, Busy: {self.engine and self.engine.isBusy()}")

            # Trigger speech if conditions are met, there are detections, and the TTS engine is ready
            if should_speak and detections and self.engine and not self.engine.isBusy():
                description = self.create_description(detections, frame_width)
                
                # Start speech in a separate daemon thread to avoid blocking the main video loop
                speech_thread = threading.Thread(target=self.speak_text, args=(description,))
                speech_thread.daemon = True 
                speech_thread.start()
                
                self.spoken_objects = current_objects_set # Update the set of spoken objects
                self.last_spoken_time = now # Reset the timer for periodic speech
            
            # --- Visual Overlay (Bounding Boxes & Text) ---
            frame = self.draw_detections(frame, detections) # Draw detected objects on the frame
            
            # Display current status and detected object names
            status_text = f"Detections: {len(detections)} | Models: {len(self.models)} | Conf: {self.confidence_threshold:.2f}"
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            if detections:
                detected_names_str = ', '.join(sorted(list(current_objects_set))) # Display unique names, sorted alphabetically
                cv2.putText(frame, f"Visible: {detected_names_str}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Visible: None", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

            # Show controls and available classes overlay based on 'h' key or initial time
            if show_info_overlay or (time.time() - self.start_time < 15): # Display for first 15 seconds or if 'h' is toggled
                info_line1 = "Controls: 'q'-Quit | 's'-Speak | 'c'-Confidence | 'h'-Toggle Info"
                sample_classes = list(self.all_classes)[:10] # Show first 10 classes as an example
                info_line2 = f"Classes: {', '.join(sample_classes)}..."
                
                cv2.putText(frame, info_line1, (10, frame.shape[0] - 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.putText(frame, info_line2, (10, frame.shape[0] - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            
            # --- DEBUG: BEFORE CV2.IMSHOW ---
            # print("DEBUG: Calling cv2.imshow()...") # Uncomment for very granular debug
            cv2.imshow('VisionSpeak Real-time Object Detection', frame)
            # --- DEBUG: AFTER CV2.IMSHOW ---
            # print("DEBUG: cv2.imshow() called.") # Uncomment for very granular debug

            # --- DEBUG: BEFORE CV2.WAITKEY ---
            # print("DEBUG: Calling cv2.waitKey(1)...") # Uncomment for very granular debug
            key = cv2.waitKey(1) & 0xFF # Wait for 1ms and get key press
            # --- DEBUG: AFTER CV2.WAITKEY ---
            # print(f"DEBUG: cv2.waitKey(1) returned key={key}") # Uncomment for very granular debug

            if key == ord('q'): # Quit
                print("DEBUG: 'q' pressed. Breaking loop.")
                break
            elif key == ord('s'): # Force immediate narration
                if detections and self.engine and not self.engine.isBusy():
                    description = self.create_description(detections, frame_width)
                    speech_thread = threading.Thread(target=self.speak_text, args=(description,))
                    speech_thread.daemon = True 
                    speech_thread.start()
                    self.spoken_objects = current_objects_set 
                    self.last_spoken_time = now 
            elif key == ord('c'): # Toggle confidence threshold
                self.confidence_threshold = 0.25 if self.confidence_threshold > 0.3 else 0.45
                print(f"Confidence threshold set to: {self.confidence_threshold:.2f}")
            elif key == ord('h'): # Toggle info overlay visibility
                show_info_overlay = not show_info_overlay
        
        # --- DEBUG: EXITING LOOP ---
        print("DEBUG: Exited main loop. Calling cleanup.")
        self.cleanup()
    
    def cleanup(self):
        """Releases camera resources, destroys OpenCV windows, and stops the TTS engine."""
        if self.cap:
            self.cap.release()
            print("Webcam released.")
        cv2.destroyAllWindows()
        print("OpenCV windows destroyed.")
        if self.engine:
            self.engine.stop() # Explicitly stop the TTS engine to release resources
            print("TTS engine stopped.")
        print("VisionSpeak stopped. All resources released.")

if __name__ == "__main__":
    detector = MultiModelDetector()
    detector.run()