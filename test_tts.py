"""
Test to demonstrate Windows TTS issue and fix
"""

import pyttsx3
import time

print("="*60)
print("Windows TTS Fix Test")
print("="*60)

# Test 1: OLD METHOD (Reusing engine) - FAILS ON WINDOWS
print("\n1️⃣ Testing OLD method (reusing same engine)...")
print("This often fails on Windows after first speech\n")

try:
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    
    messages = [
        "Message one",
        "Message two", 
        "Message three"
    ]
    
    for i, msg in enumerate(messages, 1):
        print(f"   Attempt {i}: {msg}")
        try:
            engine.say(msg)
            engine.runAndWait()
            print(f"   ✅ Success")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
        time.sleep(0.5)
    
    engine.stop()
    del engine
    
except Exception as e:
    print(f"❌ Old method failed: {e}")

print("\n" + "-"*60)
time.sleep(2)

# Test 2: NEW METHOD (Reinitialize engine) - WORKS ON WINDOWS
print("\n2️⃣ Testing NEW method (fresh engine each time)...")
print("This works reliably on Windows\n")

messages = [
    "Message one",
    "Message two",
    "Message three"
]

for i, msg in enumerate(messages, 1):
    print(f"   Attempt {i}: {msg}")
    try:
        # Create FRESH engine for each speech
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.say(msg)
        engine.runAndWait()
        engine.stop()
        del engine
        print(f"   ✅ Success")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    time.sleep(0.5)

print("\n" + "="*60)
print("Test Complete!")
print("="*60)

print("\nResults:")
print("  Method 1 (reuse): Probably only worked once ❌")
print("  Method 2 (reinit): Should work all 3 times ✅")

print("\nYour detector now uses Method 2 (Windows fix)")
print("="*60)