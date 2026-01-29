"""
YAMNet Audio Detection for Baby Monitor (TPU/Raspberry Pi Version)
"""

import numpy as np
import sounddevice as sd
import csv
import time
from collections import deque
import threading
import platform

# ============================================================================
# TPU / TENSORFLOW IMPORTS
# ============================================================================
# On Raspberry Pi, we usually use tflite_runtime. On PC, we use tensorflow.
try:
    import tflite_runtime.interpreter as tflite
    print("✓ Using tflite_runtime (Raspberry Pi mode)")
except ImportError:
    try:
        import tensorflow.lite as tflite
        print("✓ Using full tensorflow (PC mode)")
    except ImportError:
        print("Error: Please install 'tflite-runtime' or 'tensorflow'")
        exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

SAMPLE_RATE = 16000
WINDOW_SIZE = 0.975
HOP_SIZE = 0.5
CONFIDENCE_THRESHOLD = 0.3
MICROPHONE_DEVICE = None
AUDIO_GAIN = 5.0
# The specific file name for the TPU-compiled model
MODEL_FILE = 'yamnet_edgetpu.tflite' 

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================

audio_buffer = deque(maxlen=int(SAMPLE_RATE * WINDOW_SIZE))
buffer_lock = threading.Lock()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_class_names(csv_path='yamnet_class_map.csv'):
    class_names = {}
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) >= 3:
                    class_names[int(row[0])] = row[2]
    except FileNotFoundError:
        print(f"Warning: {csv_path} not found.")
        return None
    return class_names

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Audio status: {status}")
    with buffer_lock:
        audio_buffer.extend(indata[:, 0])

def get_audio_window():
    with buffer_lock:
        if len(audio_buffer) < int(SAMPLE_RATE * WINDOW_SIZE):
            return None
        audio_data = np.array(list(audio_buffer), dtype=np.float32)
        audio_data = audio_data * AUDIO_GAIN
        audio_data = np.clip(audio_data, -1.0, 1.0)
        return audio_data

def make_interpreter(model_path):
    """Creates the interpreter with Edge TPU delegate if available"""
    try:
        # Try to load the Edge TPU delegate
        # This library name 'libedgetpu.so.1' is standard for Linux/Pi
        delegate = tflite.load_delegate('libedgetpu.so.1')
        print("Edge TPU Delegate loaded successfully!")
        return tflite.Interpreter(model_path=model_path, experimental_delegates=[delegate])
    except ValueError:
        print("Could not load Edge TPU Delegate. Falling back to CPU (slower).")
        print("(Make sure libedgetpu is installed and the TPU is plugged in)")
        return tflite.Interpreter(model_path=model_path)
    except Exception as e:
        print(f"Error creating interpreter: {e}")
        return tflite.Interpreter(model_path=model_path)

def run_inference(interpreter, audio_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Check if the model expects int8 (quantized) or float32 input
    input_type = input_details[0]['dtype']
    
    # Prepare input
    audio_data = np.array(audio_data, dtype=np.float32)
    
    # If the TPU model expects int8 input (common for quantized models), convert it
    # Note: YAMNet usually takes float input even on TPU, but good to be safe.
    # For standard YAMNet, input index 0 usually expects [1, 15600] float32
    
    expected_length = input_details[0]['shape'][-1]
    if len(audio_data) != expected_length:
        if len(audio_data) < expected_length:
            audio_data = np.pad(audio_data, (0, expected_length - len(audio_data)))
        else:
            audio_data = audio_data[:expected_length]
            
    # Reshape to match input (usually [1, N])
    input_data = np.expand_dims(audio_data, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    scores = interpreter.get_tensor(output_details[0]['index'])
    return scores

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("YAMNet Audio Monitor (TPU Enabled)")
    print("=" * 80)
    
    # 1. Load Model
    print(f"Loading {MODEL_FILE}...")
    try:
        interpreter = make_interpreter(MODEL_FILE)
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load model. {e}")
        return

    # 2. Load Classes
    class_names = load_class_names()
    
    # 3. Setup Mic
    print("\nInitializing Audio...")
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        callback=audio_callback,
        blocksize=int(SAMPLE_RATE * HOP_SIZE),
        device=MICROPHONE_DEVICE
    )
    
    print("\nSystem Ready. Listening...")
    print("=" * 80)

    try:
        with stream:
            time.sleep(WINDOW_SIZE)
            while True:
                audio_data = get_audio_window()
                if audio_data is None:
                    time.sleep(0.1)
                    continue
                
                scores = run_inference(interpreter, audio_data)
                avg_scores = np.mean(scores, axis=0)
                top_indices = np.argsort(avg_scores)[-3:][::-1]
                
                # UI Output
                print("\r" + " " * 100, end="\r")
                detections = []
                for idx in top_indices:
                    confidence = avg_scores[idx]
                    if confidence > CONFIDENCE_THRESHOLD:
                        name = class_names.get(idx, f'{idx}') if class_names else f'{idx}'
                        detections.append(f"{name}: {confidence:.2f}")
                
                if detections:
                    print(f"{' | '.join(detections)}", end="", flush=True)
                else:
                    print("...", end="", flush=True)
                
                time.sleep(HOP_SIZE)
            
    except KeyboardInterrupt:
        print("\nStopping...")

if __name__ == "__main__":
    main()