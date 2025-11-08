import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time

# Import the class you created in flash.py
from flash import MultiModelDetector 

class DetectorApp:
    # --- FIX: Changed _init_ to __init__ ---
    def __init__(self, root):
        self.root = root
        root.title("VisionSpeak Detector")
        root.geometry("400x200")
        
        # Initialize your detector (Backend)
        # This will now work correctly because __init__ is spelled right
        self.detector = MultiModelDetector()
        self.detection_thread = None
        
        # Create the UI components
        self._setup_ui()
        
    def _setup_ui(self):
        # Styling and layout
        style = ttk.Style()
        style.configure('TButton', font=('Arial', 10), padding=10)
        style.configure('TLabel', font=('Arial', 10))

        # Main frame
        main_frame = ttk.Frame(self.root, padding="20 20 20 20")
        main_frame.pack(fill='both', expand=True)
        
        # Status Label
        self.status_var = tk.StringVar(value="Status: Ready")
        self.status_label = ttk.Label(main_frame, textvariable=self.status_var)
        self.status_label.pack(pady=10)

        # Start Button
        self.start_button = ttk.Button(main_frame, text="Start Detection", command=self.start_detection)
        self.start_button.pack(pady=10, fill='x')

        # Stop Button
        self.stop_button = ttk.Button(main_frame, text="Stop Detection", command=self.stop_detection, state=tk.DISABLED)
        self.stop_button.pack(pady=10, fill='x')

    def start_detection(self):
        if self.detection_thread and self.detection_thread.is_alive():
            messagebox.showinfo("Detector", "Detection is already running!")
            return

        self.status_var.set("Status: Starting...")
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        # 1. Set the control flag on the detector
        self.detector.running = True
        
        # 2. Start the detector's run() method in a new thread
        self.detection_thread = threading.Thread(target=self._run_detector_safe, daemon=True)
        self.detection_thread.start()
        
        self.status_var.set("Status: Detecting...")
        
    def stop_detection(self):
        if self.detector.running:
            self.status_var.set("Status: Stopping...")
            self.stop_button.config(state=tk.DISABLED)
            
            # 1. Gracefully stop the thread loop
            self.detector.running = False
            
            if self.detection_thread and self.detection_thread.is_alive():
                print("Waiting for detection thread to join...")
                pass
            
            # Reset UI after a small delay to allow cleanup to finish
            self.root.after(100, self._reset_ui) # Tkinter-safe way to delay

    def _reset_ui(self):
        # This function runs on the main thread
        self.status_var.set("Status: Ready")
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        print("Detector safely stopped and UI reset.")
        
    def _run_detector_safe(self):
        """Wrapper to catch exceptions and ensure UI is reset if detection crashes"""
        try:
            # This runs in the separate thread
            self.detector.run()
        except Exception as e:
            print(f"FATAL DETECTOR ERROR: {e}")
            messagebox.showerror("Error", f"Detection loop crashed: {e}")
        finally:
            # Ensure UI reset runs on the main Tkinter thread
            self.root.after(0, self._reset_ui)


# --- FIX: Changed "_main_" to "__main__" ---
if __name__ == "__main__":
    root = tk.Tk()
    app = DetectorApp(root)
    root.mainloop()