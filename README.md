# VisionSpeak Detector

VisionSpeak is a real-time object detection application that provides live audio narration. It uses a webcam to "see" the world, identifies objects in the frame, and speaks what it finds using a natural-language description.

This project uses YOLO (You Only Look Once) for state-of-the-art object detection, OpenCV for video processing, and pyttsx3 for text-to-speech narration, all wrapped in a simple desktop application.

## 🚀 Key Features

* **Real-Time Detection:** Identifies 80 different common objects (people, cups, laptops, etc.) from a live webcam feed.
* **Audio Narration:** Provides thread-safe, spoken descriptions of the objects in view (e.g., "I see a person and a laptop").
* **Multi-Model Engine:** Uses multiple YOLO models (`yolov8s.pt` and `yolov8m.pt`) to combine general-purpose and detailed detection for better accuracy.
* **Smart Filtering:**
    * **Stability Filter:** Reduces "flickering" by only announcing objects that are stable for several frames.
    * **Duplicate Removal:** Intelligently merges overlapping detections of the same object.
    * **Smart Correction:** Corrects common misdetections (e.g., a small, low-confidence "donut" is corrected to "round object").
* **Intelligent Narration:**
    * Prioritizes announcing *changes* in the scene (new objects appearing, objects disappearing).
    * Provides periodic updates even if the scene is static.

## 🔧 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

* Python 3.8 or newer
* A webcam connected to your computer

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/chinmayicyi/VisionSpeak.git](https://github.com/chinmayicyi/VisionSpeak.git)
    cd VisionSpeak
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```
    ```bash
    # macOS / Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The YOLO `.pt` model files will be downloaded automatically by the application the first time you run it.*

## 🏃‍♂️ How to Run

This project runs as a simple desktop application.

1.  Ensure you have installed the requirements (see "Installation" above).
2.  Run the main application file from your terminal:

    ```bash
    python app.py
    ```
3.  A window will appear. Click the **"Start Detection"** button to begin the webcam feed and narration.

## 🛠️ Core Technologies Used

* [**Ultralytics YOLOv8**](https://ultralytics.com/): The core real-time object detection model.
* [**OpenCV**](https://opencv.org/): Used for webcam capture, video processing, and drawing visuals.
* [**pyttsx3**](https://pyttsx3.readthedocs.io/en/latest/): A text-to-speech library for offline audio narration.
* [**Tkinter**](https://docs.python.org/3/library/tkinter.html): Used to create the simple desktop GUI.