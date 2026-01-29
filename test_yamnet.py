"""
YAMNet Audio Detection for Baby Monitor
Live audio monitoring with real-time sound classification
"""

import numpy as np
import sounddevice as sd
import tensorflow as tf
import csv
import time
from collections import deque
import threading

# ============================================================================
# CONFIGURATION - Adjust these settings as needed
# ============================================================================

SAMPLE_RATE = 16000               # Audio sample rate (Hz)
WINDOW_SIZE = 0.975               # Audio window size (seconds)
HOP_SIZE = 0.5                    # How often to run inference (seconds)
CONFIDENCE_THRESHOLD = 0.3        # Minimum confidence to display detections
MICROPHONE_DEVICE = 2         # Microphone device index (None = default)
AUDIO_GAIN = 5.0                 # Amplification factor (1.0 = no change, 5.0 = 5x louder)

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================

audio_buffer = deque(maxlen=int(SAMPLE_RATE * WINDOW_SIZE))
buffer_lock = threading.Lock()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_class_names(csv_path='yamnet_class_map.csv'):
    """Load YAMNet class names from CSV file"""
    class_names = {}
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 3:
                    class_names[int(row[0])] = row[2]  # index -> display_name
    except FileNotFoundError:
        print(f"Warning: {csv_path} not found. Using class indices instead.")
        return None
    return class_names


def audio_callback(indata, frames, time_info, status):
    """Callback function for continuous audio stream"""
    if status:
        print(f"Audio status: {status}")
    
    with buffer_lock:
        audio_buffer.extend(indata[:, 0])


def get_audio_window():
    """Get current audio window from buffer with gain applied"""
    with buffer_lock:
        if len(audio_buffer) < int(SAMPLE_RATE * WINDOW_SIZE):
            return None
        audio_data = np.array(list(audio_buffer), dtype=np.float32)
        
        # Apply gain to amplify quiet microphones
        audio_data = audio_data * AUDIO_GAIN
        
        # Clip to prevent distortion
        audio_data = np.clip(audio_data, -1.0, 1.0)
        
        return audio_data


def run_inference(interpreter, audio_data):
    """Run YAMNet inference on audio data"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Prepare input - YAMNet TFLite expects 1D array
    audio_data = np.array(audio_data, dtype=np.float32)
    
    # Ensure correct length
    expected_length = input_details[0]['shape'][-1] if len(input_details[0]['shape']) > 0 else 15600
    if len(audio_data) != expected_length:
        if len(audio_data) < expected_length:
            audio_data = np.pad(audio_data, (0, expected_length - len(audio_data)))
        else:
            audio_data = audio_data[:expected_length]
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], audio_data)
    interpreter.invoke()
    
    # Get output scores
    scores = interpreter.get_tensor(output_details[0]['index'])
    
    return scores

# ============================================================================
# MAIN DETECTION LOOP
# ============================================================================

def main():
    print("=" * 80)
    print("YAMNet Live Audio Detection - Baby Monitor")
    print("=" * 80)
    
    # Load model
    model_path = 'yamnet.tflite'
    print(f"\nLoading model from {model_path}...")

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    print("✓ Model loaded successfully!")
    
    
    # Load class names
    class_names = load_class_names()
    if class_names:
        print("✓ Class names loaded")
    else:
        print("⚠ Running without class names (will show class indices)")
    
    # Display audio devices
    print("\nAvailable input devices:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            default_marker = " (DEFAULT)" if i == sd.default.device[0] else ""
            print(f"  [{i}] {device['name']}{default_marker}")
    
    if MICROPHONE_DEVICE is not None:
        print(f"\nUsing microphone: [{MICROPHONE_DEVICE}] {devices[MICROPHONE_DEVICE]['name']}")
    else:
        print(f"\nUsing default microphone")
    
    print("\n" + "=" * 80)
    print("Starting live audio monitoring...")
    print(f"Configuration: Gain={AUDIO_GAIN}x, Threshold={CONFIDENCE_THRESHOLD}")
    print("Press Ctrl+C to stop")
    print("=" * 80 + "\n")
    
    # Start audio stream
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        callback=audio_callback,
        blocksize=int(SAMPLE_RATE * HOP_SIZE),
        device=MICROPHONE_DEVICE
    )
    
    try:
        with stream:
            print("🎤 Listening...\n")
            time.sleep(WINDOW_SIZE)
            
            while True:
                # Get audio window
                audio_data = get_audio_window()
                
                if audio_data is None:
                    time.sleep(0.1)
                    continue
                
                # Run inference
                scores = run_inference(interpreter, audio_data)
                avg_scores = np.mean(scores, axis=0)
                
                # Get top predictions
                top_indices = np.argsort(avg_scores)[-3:][::-1]
                
                # Display compact output
                print("\r" + " " * 100, end="\r")
                
                detections = []
                for idx in top_indices:
                    confidence = avg_scores[idx]
                    if confidence > CONFIDENCE_THRESHOLD:
                        name = class_names.get(idx, f'Class {idx}') if class_names else f'Class {idx}'
                        detections.append(f"{name}: {confidence:.2f}")
                
                if detections:
                    print(f"🔊 {' | '.join(detections)}", end="", flush=True)
                else:
                    print("🔇 [Below threshold]", end="", flush=True)
                
                # Wait before next inference
                time.sleep(HOP_SIZE)
            
    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("Stopping audio monitoring...")
        print("=" * 80)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()