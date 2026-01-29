INSTRUCTIONS FOR RASPBERRY PI + CORAL TPU SETUP
===============================================

FILES IN THIS PACKAGE:
1. yamnet_tpu.py        <- RUN THIS SCRIPT (Main logic)
2. yamnet.tflite        <- Standard model (for CPU testing)
3. yamnet_class_map.csv <- Labels
4. requirements.txt     <- Python libraries


STEP 1: SYSTEM DEPENDENCIES (Run in Terminal)
You must install the audio driver and TPU runtime before Python will work.
$ sudo apt-get update
$ sudo apt-get install libportaudio2
$ echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
$ curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
$ sudo apt-get update
$ sudo apt-get install libedgetpu1-std

STEP 2: PYTHON SETUP
$ pip3 install -r requirements.txt
$ pip3 install tflite-runtime

STEP 3: COMPILE THE MODEL
The 'yamnet.tflite' included here is for CPU. 
To use the TPU, you must compile it or download the pre-compiled version.
1. Install the compiler: https://coral.ai/docs/edgetpu/compiler/
2. Run: edgetpu_compiler yamnet.tflite
3. This creates 'yamnet_edgetpu.tflite'.
4. Ensure 'yamnet_edgetpu.tflite' is in the same folder as the script.

STEP 4: RUN IT
$ python3 yamnet_tpu.py

It should detect audio and classify it in real-time.