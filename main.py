from tools import Realtime, get_audio_input_stream
import pyaudio
# import model
from queue import Queue
from threading import Thread
from settings import Settings
import sys
import time
# import create_training_examples
import numpy as np
import os
import tensorflow as tf
from keras.models import load_model
# from settings import Settings
tf.compat.v1.disable_v2_behavior()
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

chunk_duration = 0.5
silence_threshold = 100
fs = 44100 
chunk_samples = int(fs * chunk_duration)
feed_duration = 10
feed_samples = int(fs * feed_duration)

model = load_model('models/tr_model.h5')
if model:
    print("YESS")
else:
    print('TATATATA byeee')

settings = Settings()
realtime = Realtime(settings,model)

# realtime.lastcall()


q = Queue()

run = True

silence_threshold = 100

# Run the demo for a timeout seconds
timeout = time.time() + 0.5*60  # 0.5 minutes from now

# Data buffer for the input wavform
data = np.zeros(feed_samples, dtype='int16')

def callback(in_data, frame_count, time_info, status):
    global run, timeout, data, silence_threshold    
    if time.time() > timeout:
        run = False        
    data0 = np.frombuffer(in_data, dtype='int16')
    if np.abs(data0).mean() < silence_threshold:
        # sys.stdout.write('-')
        print('-')
        return (in_data, pyaudio.paContinue)
    else:
        # sys.stdout.write('.')
        print('.')
    data = np.append(data,data0)    
    if len(data) > feed_samples:
        data = data[-feed_samples:]
        # Process data async by sending a queue.
        q.put(data)
    return (in_data, pyaudio.paContinue)

stream = get_audio_input_stream(callback)
stream.start_stream()


try:
    while run:
        data = q.get()
        # print("data receiving!")
        spectrum = realtime.get_spectrogram(data)
        preds = realtime.detect_triggerword_spectrum(spectrum)
        new_trigger = realtime.has_new_triggerword(preds, chunk_duration, feed_duration)
        # print("predicting...")
        if new_trigger:
            sys.stdout.write('1')
except (KeyboardInterrupt, SystemExit):
    stream.stop_stream()
    stream.close()
    timeout = time.time()
    run = False
        
stream.stop_stream()
stream.close()




