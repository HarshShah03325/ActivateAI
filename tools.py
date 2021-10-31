from queue import Queue
import numpy as np
import pyaudio
import matplotlib.mlab as mlab
import time
import sys
from queue import Queue
from pydub import AudioSegment
from pydub.playback import play

timeout = time.time() + 30
chunk_duration = 0.5
silence_threshold = 100
fs = 44100 
chunk_samples = int(fs * chunk_duration)
feed_duration = 10
feed_samples = int(fs * feed_duration)
data = np.zeros(feed_samples, dtype='int16')
q = Queue()
run = True

class Realtime:
    def __init__(self,settings, model):
        self.Tx = settings.Tx
        self.Ty = settings.Ty
        self.n_freq = settings.n_freq
        self.model = model


    def detect_triggerword_spectrum(self,x):
        """
        Function to predict the location of the trigger word.
        
        Argument:
        x -- spectrum of shape (freqs, Tx)
        i.e. (Number of frequencies, The number time steps)

        Returns:
        predictions -- flattened numpy array to shape (number of output time steps)
        """
        # the spectogram outputs  and we want (Tx, freqs) to input into the model
        x  = x.swapaxes(0,1)
        x = np.expand_dims(x, axis=0)
        predictions = self.model.predict(x)
        return predictions.reshape(-1)

    def has_new_triggerword(self,predictions, chunk_duration, feed_duration, threshold=0.4):
        """
        Function to detect new trigger word in the latest chunk of input audio.
        It is looking for the rising edge of the predictions data belongs to the
        last/latest chunk.
        
        Argument:
        predictions -- predicted labels from model
        chunk_duration -- time in second of a chunk
        feed_duration -- time in second of the input to model
        threshold -- threshold for probability above a certain to be considered positive

        Returns:
        True if new trigger word detected in the latest chunk
        """
        predictions = predictions > threshold
        chunk_predictions_samples = int(len(predictions) * chunk_duration / feed_duration)
        chunk_predictions = predictions[-chunk_predictions_samples:]
        level = chunk_predictions[0]
        for pred in chunk_predictions:
            if pred > level:
                return True
            else:
                level = pred
        return False

    def get_spectrogram(self,data):
        """
        Function to compute spectrogram.

        Arguments:
        data -- stream of np.array after processing the raw audio data

        Returns:
        pxx - computed spectrogram of the data (2D np.array)
        """
        nfft = 200 # Length of each window segment
        fs = 8000 # Sampling frequencies
        noverlap = 120 # Overlap between windows
        nchannels = data.ndim
        if nchannels == 1:
            pxx, _, _ = mlab.specgram(data, nfft, fs, noverlap = noverlap)
        elif nchannels == 2:
            pxx, _, _ = mlab.specgram(data[:,0], nfft, fs, noverlap = noverlap)
        return pxx

    
    def callback(self,in_data, frame_count, time_info, status):
        """
        Function that enables real-time prediction of the audio using a queue. Compares the mean of data with silence threshold and
        appends it to queue if greater.

        Arguments:
        in_data -- input from the audio format(np.array).

        Returns:
        in_data -- same as input data, used to fill in the queue.
        pyaudio.paContinue -- signal to continue the callback function.
        """
        global run, timeout, data, silence_threshold    
        if time.time() > timeout:
            run = False        
        data0 = np.frombuffer(in_data, dtype='int16')
        if np.abs(data0).mean() < silence_threshold:
            sys.stdout.write('-')
            return (in_data, pyaudio.paContinue)
        else:
            sys.stdout.write('.')
        data = np.append(data,data0)    
        if len(data) > feed_samples:
            data = data[-feed_samples:]
            # Process data async by sending a queue.
            q.put(data)
        return (in_data, pyaudio.paContinue)


    def run(self,callback):
        """
        Function that runs on stream of input data using callback function and predicts the presence of new trigger word.
        Produces a chiming sound if detected.

        Arguments:
        callback function
        """
        global run,q,chunk_duration,feed_duration,data
        stream = self.get_audio_input_stream(callback)
        stream.start_stream()
        try:
            while run:
                data = q.get()
                spectrum = self.get_spectrogram(data)
                preds = self.detect_triggerword_spectrum(spectrum)
                new_trigger = self.has_new_triggerword(preds, chunk_duration, feed_duration)
                if new_trigger:
                    sys.stdout.write('Trigger word detected!')
                    sound = AudioSegment.from_wav('chime.wav')
                    play(sound)
                    sys.exit()
        except (KeyboardInterrupt, SystemExit):
            stream.stop_stream()
            stream.close()
            timeout = time.time()
            run = False
        stream.stop_stream()
        stream.close()

    
    def get_audio_input_stream(self,callback):
        """
        Function to get stream of audio data from the device using pyaudio library.

        Arguments:
        callback function to be called on the stream.

        Returns:
        stream of audio data in frames.
        """
        stream = pyaudio.PyAudio().open(
            format=pyaudio.paInt16,
            channels=1,
            rate=fs,
            input=True,
            frames_per_buffer=chunk_samples,
            input_device_index=0,
            stream_callback=callback)
        return stream
