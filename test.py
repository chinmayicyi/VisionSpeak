import pyttsx3
import time

try:
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    print("Test: Initializing TTS engine...")

    print("Test: Speaking 'Hello, this is a test from VisionSpeak.'")
    engine.say("Hello, this is a test from VisionSpeak.")
    engine.runAndWait()
    print("Test: Finished speaking.")

    # Check if the engine is still busy after runAndWait (shouldn't be)
    if engine.isBusy():
        print("Test: Warning - Engine is still busy after runAndWait().")
    
    # Try speaking again immediately without waiting to test busy state
    print("Test: Speaking 'Second test' quickly.")
    engine.say("Second test")
    engine.runAndWait() # This will block until it finishes
    print("Test: Second test finished.")

except Exception as e:
    print(f"Test: An error occurred: {e}")

finally:
    if 'engine' in locals() and engine:
        engine.stop()
        print("Test: TTS engine stopped.")