import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import cv2 # Import cv2

# Import the class from our newly merged flash.py
from flash import MultiModelDetector 

class DetectorApp:
    def __init__(self, root):
        self.root = root
        root.title("VisionSpeak Detector")
        root.geometry("400x250") # Made window a little taller
        
        # Detector is now created *when we click start*
        self.detector = None
        self.detection_thread = None
        
        self._setup_ui()
        
        # Make sure cleanup happens when window is closed
        root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def _setup_ui(self):
        style = ttk.Style()
        style.configure('TButton', font=('Arial', 10), padding=10)
        style.configure('TLabel', font=('Arial', 10))

        main_frame = ttk.Frame(self.root, padding="20 20 20 20")
        main_frame.pack(fill='both', expand=True)
        
        self.status_var = tk.StringVar(value="Status: Ready")
        self.status_label = ttk.Label(main_frame, textvariable=self.status_var)
        self.status_label.pack(pady=10)

        self.start_button = ttk.Button(main_frame, text="Start Detection", command=self.start_detection)
        self.start_button.pack(pady=10, fill='x')

        self.stop_button = ttk.Button(main_frame, text="Stop Detection", command=self.stop_detection, state=tk.DISABLED)
        self.stop_button.pack(pady=10, fill='x')
        
        # --- NEW: Pause Button ---
        self.pause_button = ttk.Button(main_frame, text="Pause/Resume Narration", command=self.toggle_pause, state=tk.DISABLED)
        self.pause_button.pack(pady=10, fill='x')

    def start_detection(self):
        if self.detection_thread and self.detection_thread.is_alive():
            messagebox.showinfo("Detector", "Detection is already running!")
            return

        self.status_var.set("Status: Starting... Loading models...")
        self.start_button.config(state=tk.DISABLED)
        
        # --- NEW: Create detector instance on-demand ---
        # This ensures all resources (like camera) are fresh
        self.detector = MultiModelDetector()
        
        if not self.detector.cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open webcam. Is it in use?")
            self._reset_ui()
            return

        self.detector.running = True
        
        self.detection_thread = threading.Thread(target=self._run_detector_safe, daemon=True)
        self.detection_thread.start()
        
        self.status_var.set("Status: Detecting...")
        self.stop_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.NORMAL)
        
    def stop_detection(self):
        if self.detector and self.detector.running:
            self.status_var.set("Status: Stopping...")
            self.stop_button.config(state=tk.DISABLED)
            self.pause_button.config(state=tk.DISABLED)
            
            # Set flag to stop the loop
            self.detector.running = False
            
            # Wait for thread to finish (it will call cleanup)
            if self.detection_thread and self.detection_thread.is_alive():
                print("Waiting for detection thread to join...")
                # We don't need to join, the _reset_ui will be called from the thread
                pass
        
        # Use root.after to let the UI update before reset
        self.root.after(100, self._reset_ui)

    def _reset_ui(self):
        self.status_var.set("Status: Ready")
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.DISABLED)
        
        # Clean up the detector instance
        self.detector = None
        self.detection_thread = None
        
        # --- NEW: Explicitly destroy OpenCV windows ---
        cv2.destroyAllWindows()
        print("Detector safely stopped and UI reset.")
        
    def _run_detector_safe(self):
        """Wrapper to catch exceptions and ensure UI is reset if detection crashes"""
        try:
            self.detector.run()
        except Exception as e:
            print(f"FATAL DETECTOR ERROR: {e}")
            messagebox.showerror("Error", f"Detection loop crashed: {e}")
        finally:
            # Ensure UI reset runs on the main Tkinter thread
            self.root.after(0, self._reset_ui)

    # --- NEW: Function for the pause button ---
    def toggle_pause(self):
        if self.detector:
            self.detector.narration_paused = not self.detector.narration_paused
            status = "paused" if self.detector.narration_paused else "resumed"
            print(f"Narration {status}")
            self.status_var.set(f"Status: Detecting... (Narration {status})")

    # --- NEW: Handle window close button ---
    def on_closing(self):
        if self.detector and self.detector.running:
            self.stop_detection()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = DetectorApp(root)
    root.mainloop()