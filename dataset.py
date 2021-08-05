import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
from pydub import AudioSegment
import numpy as np


class Dataset:
    def __init__(self, settings):
        self.wake_sound = settings.wake_sound
        self.Ty = settings.Ty
        self.m = settings.m
        self.positives = []
        self.negatives = []
        self.backgrounds = []
        self.X_train = []
        self.Y_train = []
        self.X_dev = []
        self.Y_dev = []

    def graph_spectogram(self,wav_file,plotting=False):
        rate, data = wavfile.read(wav_file)
        nfft = 200 # Length of each window segment
        fs = 8000 # Sampling frequencies
        noverlap = 120 # Overlap between windows
        nchannels = data.ndim
        if nchannels == 1:
            pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
        elif nchannels == 2:
            pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
        else:
            print("The audio has more than 2 channels")

        if plotting==True:
            plt.show(block=False)
            plt.pause(0.001)

        return pxx

    
    # Used to standardize volume of audio clip
    def match_target_amplitude(self, sound, target_dBFS):
        change_in_dBFS = target_dBFS - sound.dBFS
        return sound.apply_gain(change_in_dBFS)

    
    #Loads up the raw activates, negatives and backgrounds from our data
    def load_raw(self, path):
        for filename in os.listdir(path + "activates"):
            if filename.endswith("wav"):
                positve = AudioSegment.from_wav(path + "activates/" + filename)
                self.positives.append(positve)
        for filename in os.listdir(path + "backgrounds"):
            if filename.endswith("wav"):
                background = AudioSegment.from_wav(path + "backgrounds/" + filename)
                self.backgrounds.append(background)
        for filename in os.listdir(path + "negatives"):
            if filename.endswith("wav"):
                negative = AudioSegment.from_wav(path + "negatives/" + filename)
                self.negatives.append(negative)

    
    def get_random_time_segment(self, segment_ms):
        segment_start = np.random.randint(low=0, high=10000-segment_ms)   
        segment_end = segment_start + segment_ms - 1
        return (segment_start, segment_end)


    #Check whether the present segment overlaps with the previous segments
    def is_overlapping(self, segment, previous_segments):
        segment_start, segment_end = segment
        overlap=False
        for previous_start, previous_end in previous_segments:
            if segment_start <= previous_end and previous_start <= segment_end:
                overlap = True
        return overlap

    # Insert audio clip(positive, negative or background) such that it does not overlap  
    def insert_audio_clip(self, background, audio_clip, previous_segments):
        
        segment_ms = len(audio_clip)
        segment_time = self.get_random_time_segment(segment_ms)
    
        retry = 5
        while self.is_overlapping(segment_time, previous_segments) and retry >= 0:
            segment_time = self.get_random_time_segment(segment_ms)
            retry = retry - 1
        if not self.is_overlapping(segment_time, previous_segments):
            previous_segments.append(segment_time)
            new_background = background.overlay(audio_clip, position = segment_time[0])
        else:
            new_background = background
            segment_time = (10000, 10000)
        
        return new_background, segment_time

    # Insert ones in the output for next 50 timestamps
    def insert_ones(y, segment_end_ms):
        _, Ty = y.shape
        segment_end_y = int(segment_end_ms * Ty / 10000.0)
        if segment_end_y < Ty:
            for i in range(segment_end_y+1, segment_end_y+51):
                if i < Ty:
                    y[0, i] = 1
        return y


    def create_single_example(self, background,i):
        background = background-20
        y = np.zeors((1,self.Ty))
        previous_segments = []


        number_of_positives = np.random.randint(low=0, high=5) #Randomly chooses the number of positives in an example 
        random_indices = np.random.randint(low=0, high=len(self.positives), size=number_of_positives)  #randomly chooses an index from the number of positives
        random_positives = [self.positives[i] for i in random_indices]

        for random_positive in random_positives:
            background, segment_time = self.insert_audio_clip(
                background, random_positive, previous_segments)
            segment_start, segment_end = segment_time
            self.insert_ones(y,segment_end)

        number_of_negatives = np.random.randint(low=0, high=3)
        random_indices = np.random.randint(low=0, high=len(self.negatives), size = number_of_negatives)
        random_negatives = [self.negatives[i] for i in random_indices]

        for random_negative in random_negatives:
            background, _ = self.insert_audio_clip(
                background, random_negative, previous_segments)

        background = self.match_target_amplitude(background, -20.0)
        file_handle = background.export(
            './dataset/' + self.wake_sound + '/train/train' + str(i) + '.wav', format='wav')
        print('File (train' + str(i) + '.wav) was saved in your directory.')
        x = self.graph_spectrogram(
            wav_file='./dataset/' + self.wake_sound + '/train/train' + str(i) + '.wav', plotting=False)

        return x, y


    def create_training_data(self):
        self.load_raw()

        np.random.seed(4563)
        for i in range(self.m):
            if i%10==0:
                print(str(i)+" Examples created")
            x, y = self.create_training_example(self.backgrounds(self.backgrounds[i%2], i))
            self.X_train.append(x.swapaxes(0,1))
            self.Y_train.append(y.swapaxes(0,1))
        X = np.array(self.X_train)
        Y = np.array(self.Y_train)

        np.save(f'./data/train_XY/X.npy',X)
        np.save(f'./data/train_XY/Y.npy',Y)

    def load_dataset(self):
        self.X_train = np.load('./XY_train/X.npy')
        self.Y_train = np.load('./XY_train/Y.npy')
        self.X_dev = np.load('./XY_dev/X_dev.npy')
        self.Y_dev = np.load('./XY_dev/Y_dev.npy')

        

