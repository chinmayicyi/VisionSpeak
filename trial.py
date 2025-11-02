import cv2
from ultralytics import YOLO
import numpy as np
import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Load the YOLO model
yolo = YOLO('yolov8s.pt')

# Function to generate distinct colors for different classes
def get_colors(class_num):
    base_colors = [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255)   # Yellow
    ]
    return base_colors[class_num % len(base_colors)]

# Initialize webcam
cap = cv2.VideoCapture(0)

# Set webcam resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Dictionary to track announced objects
announced_objects = {}

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = yolo(frame)[0]
    
    # Clear previous announcements
    current_objects = set()

    # Process detections
    for detection in results.boxes.data.tolist():
        # Get detection information
        x1, y1, x2, y2, confidence, class_id = detection
        
        if confidence < 0.5:  # Skip low confidence detections
            continue

        # Convert coordinates to integers
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
        # Get class name
        class_name = results.names[int(class_id)]
        
        # Get color for this class
        color = get_colors(int(class_id))
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Add label
        label = f'{class_name} {confidence:.2f}'
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Track objects for announcement
        current_objects.add(class_name)
        
        # Announce new objects
        if class_name not in announced_objects or announced_objects[class_name] < 0:
            engine.say(f"Detected {class_name}")
            engine.runAndWait()
            announced_objects[class_name] = 10  # Wait 10 frames before announcing again
    
    # Update announcement cooldowns
    for obj in announced_objects:
        announced_objects[obj] -= 1

    # Display the frame
    cv2.imshow('Object Detection', frame)

    # Break loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()