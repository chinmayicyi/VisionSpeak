# VisionSpeak Detector

VisionSpeak is a real-time object detection application that provides live audio narration for a webcam feed. It uses a webcam to "see" the world, identifies objects in the frame, and speaks what it finds using a natural-language description.

This project uses YOLO (You Only Look Once) for state-of-the-art object detection, OpenCV for video processing, and pyttsx3 for text-to-speech narration.

## 🚀 Key Features

* **Real-Time Detection:** Identifies 80 different common objects (people, cups, laptops, etc.) from a live webcam feed.
* **Audio Narration:** Provides thread-safe, spoken descriptions of the objects in view (e.g., "I see a person and a laptop").
* **Multi-Model Engine:** Uses multiple YOLO models (`yolov8s.pt` and `yolov8m.pt`) to combine general-purpose and detailed detection for better accuracy.
* **Smart Filtering:**
    * **Stability Filter:** Reduces "flickering" by only announcing objects that are stable for several frames.
    * **Duplicate Removal:**Intelligently merges overlapping detections of the same object.
    * **Smart Correction:** Corrects common misdetections (e.g., a small, low-confidence "donut" is corrected to "round object").
* **Intelligent Narration:**
    * Prioritizes announcing *changes* in the scene (new objects appearing, objects disappearing).
    * Detects and announces object *movement*.
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
    The `ultralytics` package will automatically install PyTorch and other core dependencies.
    ```bash
    pip install -r requirements.txt
    ```
    *(If your `requirements.txt` is missing `ultralytics`, install it manually)*
    ```bash
    pip install ultralytics opencv-python pyttsx3 numpy
    ```

## 🏃‍♂️ How to Run

There are several ways to run the application. The simplest is to run the main standalone script.

### Standalone Mode (Recommended)

Run the `wtf.py` (or `works.py`) script directly from your terminal. This will open a new window showing your webcam feed with detection boxes.

```bash
python wtf.py
```

**Keyboard Controls (while the video window is active):**

* `q` - Quit the application
* `s` - Force speak the current scene immediately
* `r` - Reset tracking and force immediate narration
* `c` - Cycle through different confidence thresholds
* `SPACE` - Pause or resume live narration
* `h` - Toggle the help display on/off

### Web Application Mode

This project also includes a version that runs as a web server, allowing you to view the video stream from any device on your network.

1.  **Install web dependencies:**
    ```bash
    pip install Flask Flask-Sock waitress
    ```

2.  **Run the web server:**
    ```bash
    python web_app.py
    ```

3.  **Open your browser** and go to `http://127.0.0.1:5000` or `http://[YOUR_IP_ADDRESS]:5000` to see the feed.

## 🛠️ Core Technologies Used

* [**Ultralytics YOLOv8**](https://ultralytics.com/): The core real-time object detection model.
* [**OpenCV**](https://opencv.org/): Used for webcam capture, video processing, and drawing visuals.
* [**pyttsx3**](https://pyttsx3.readthedocs.io/en/latest/): A text-to-speech library for offline audio narration.
* [**Flask**](https://flask.palletsprojects.com/): (Optional) Used to create the web application and video stream.
* [**Tkinter**](https://docs.python.org/3/library/tkinter.html): (Optional) Used in `app.py` for a simple desktop GUI.