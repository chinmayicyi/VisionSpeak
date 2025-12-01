import tkinter as tk
from tkinter import ttk, messagebox
import threading
import cv2
import queue
from flash import MultiModelDetector

class DetectorApp:
    def __init__(self, root):
        self.root = root
        root.title("VisionSpeak Detector")
        root.geometry("500x400")
        root.resizable(False, False)

        self.detector = None
        self.detection_thread = None
        self.display_queue = queue.Queue(maxsize=1)
        self.is_displaying = False

        self._setup_ui()
        root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Start main-thread display updater
        self._update_display()

    def _setup_ui(self):
        """Setup the user interface"""
        style = ttk.Style()
        style.theme_use('clam')  # Modern theme
        style.configure('TButton', font=('Arial', 11), padding=10)
        style.configure('TLabel', font=('Arial', 10))
        style.configure('Header.TLabel', font=('Arial', 14, 'bold'))
        style.configure('Status.TLabel', font=('Arial', 10), foreground='#2e7d32')

        # Main container
        main_frame = ttk.Frame(self.root, padding="25 25 25 25")
        main_frame.pack(fill='both', expand=True)

        # Header
        header_label = ttk.Label(main_frame, text="🎥 VisionSpeak Object Detector", 
                                 style='Header.TLabel')
        header_label.pack(pady=(0, 20))

        # Status section
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="10")
        status_frame.pack(fill='x', pady=(0, 20))

        self.status_var = tk.StringVar(value="Ready to start")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, 
                                      style='Status.TLabel')
        self.status_label.pack()

        self.objects_var = tk.StringVar(value="Objects detected: 0")
        self.objects_label = ttk.Label(status_frame, textvariable=self.objects_var)
        self.objects_label.pack(pady=(5, 0))

        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x', pady=(0, 10))

        self.start_button = ttk.Button(button_frame, text="▶ Start Detection", 
                                       command=self.start_detection)
        self.start_button.pack(pady=5, fill='x')

        self.stop_button = ttk.Button(button_frame, text="⏹ Stop Detection", 
                                      command=self.stop_detection, state=tk.DISABLED)
        self.stop_button.pack(pady=5, fill='x')

        self.pause_button = ttk.Button(button_frame, text="⏸ Pause/Resume Narration", 
                                       command=self.toggle_pause, state=tk.DISABLED)
        self.pause_button.pack(pady=5, fill='x')

        self.force_speak_button = ttk.Button(button_frame, text="🔊 Force Speak Now", 
                                            command=self.force_speak, state=tk.DISABLED)
        self.force_speak_button.pack(pady=5, fill='x')

        # Info section
        info_frame = ttk.LabelFrame(main_frame, text="Information", padding="10")
        info_frame.pack(fill='x', pady=(10, 0))

        info_text = """
• Detects objects in real-time
• Narrates scene changes and movement
• Press 'q' in video window to stop
• Close window or click Stop to exit
        """
        info_label = ttk.Label(info_frame, text=info_text.strip(), 
                              justify=tk.LEFT, font=('Arial', 9))
        info_label.pack()

    def start_detection(self):
        """Start the detection process"""
        if self.detection_thread and self.detection_thread.is_alive():
            messagebox.showinfo("Detector", "Detection is already running!")
            return

        self.status_var.set("Loading models...")
        self.objects_var.set("Objects detected: 0")
        self.start_button.config(state=tk.DISABLED)
        self.root.update()

        try:
            # Initialize detector
            self.detector = MultiModelDetector(display_queue=self.display_queue)
            self.detector.running = True
            
            # Start detection thread
            self.detection_thread = threading.Thread(target=self._run_detector_safe, daemon=True)
            self.detection_thread.start()

            self.status_var.set("Detecting... (Active)")
            self.stop_button.config(state=tk.NORMAL)
            self.pause_button.config(state=tk.NORMAL)
            self.force_speak_button.config(state=tk.NORMAL)
            self.is_displaying = True

            print("✅ Detection started successfully!")

        except RuntimeError as e:
            messagebox.showerror("Camera Error", str(e))
            self._reset_ui()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start detection: {e}")
            self._reset_ui()

    def stop_detection(self):
        """Stop the detection process"""
        if self.detector and self.detector.running:
            self.status_var.set("Stopping...")
            self.stop_button.config(state=tk.DISABLED)
            self.pause_button.config(state=tk.DISABLED)
            self.force_speak_button.config(state=tk.DISABLED)

            self.detector.running = False
            self.is_displaying = False

            # Give it time to clean up
            self.root.after(200, self._reset_ui)

    def _reset_ui(self):
        """Reset UI to initial state"""
        self.status_var.set("Ready to start")
        self.objects_var.set("Objects detected: 0")
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.DISABLED)
        self.force_speak_button.config(state=tk.DISABLED)

        self.detector = None
        self.detection_thread = None
        self.is_displaying = False

        cv2.destroyAllWindows()
        print("✅ Detector stopped and UI reset")

    def _run_detector_safe(self):
        """Run detector with error handling"""
        try:
            self.detector.run()
        except Exception as e:
            print(f"❌ DETECTOR ERROR: {e}")
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: messagebox.showerror("Error", f"Detection crashed: {e}"))
        finally:
            self.root.after(0, self._reset_ui)

    def toggle_pause(self):
        """Toggle narration pause state"""
        if self.detector:
            self.detector.narration_paused = not self.detector.narration_paused
            status = "PAUSED" if self.detector.narration_paused else "Active"
            self.status_var.set(f"Detecting... ({status})")
            print(f"🔇 Narration {status.lower()}")

    def force_speak(self):
        """Force immediate narration of current scene"""
        if self.detector and not self.detector.speaking:
            # Get current objects from last detection
            current_objects = list(self.detector.last_objects)
            if current_objects:
                if len(current_objects) == 1:
                    text = f"I can see a {current_objects[0]}"
                elif len(current_objects) == 2:
                    text = f"I can see a {current_objects[0]} and a {current_objects[1]}"
                else:
                    text = f"I can see {', '.join(current_objects[:-1])}, and a {current_objects[-1]}"
                
                self.detector._speak_async(text)
                print(f"🔧 Forced narration: {text}")
            else:
                self.detector._speak_async("No objects currently detected")
                print("🔧 Forced narration: No objects")

    def on_closing(self):
        """Handle window close event"""
        if self.detector and self.detector.running:
            if messagebox.askokcancel("Quit", "Detection is running. Stop and quit?"):
                self.stop_detection()
                self.root.after(300, self.root.destroy)
        else:
            self.root.destroy()

    def _update_display(self):
        """Update display in main thread (runs continuously)"""
        if self.is_displaying and self.detector and self.detector.running:
            try:
                # Get frame from queue
                frame = self.display_queue.get_nowait()
                
                # Show frame
                cv2.imshow("VisionSpeak - Object Detection", frame)
                
                # Update object count in GUI
                if hasattr(self.detector, 'last_objects'):
                    obj_count = len(self.detector.last_objects)
                    self.objects_var.set(f"Objects detected: {obj_count}")
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("'q' pressed - stopping detection")
                    self.stop_detection()
                
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Display error: {e}")
        
        # Schedule next update
        self.root.after(30, self._update_display)


def main():
    """Main entry point"""
    print("="*50)
    print("VisionSpeak Object Detector")
    print("="*50)
    
    root = tk.Tk()
    app = DetectorApp(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    finally:
        cv2.destroyAllWindows()
        print("👋 Application closed")


if __name__ == "__main__":
    main()