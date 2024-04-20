import pyaudio
import wave
import sys
import time
import threading
import os
from collections import deque
import librosa
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# SETUP AN AUDIO RECORDER
class AudioRecorder:
    def __init__(self, filename='recording.wav', chunk=1024, channels=1, rate=44100, threshold=2000, duration=2):
        self.filename = filename
        self.chunk = chunk
        self.channels = channels
        self.rate = rate
        self.frames = []
        self.threshold = threshold
        self.duration = duration
        self.model = load_model('cnn_attempt_98_acc.h5')
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(format=pyaudio.paInt16,
                                   channels=self.channels,
                                   rate=self.rate,
                                   input=True,
                                   frames_per_buffer=self.chunk)
        self.lock = threading.Lock()
        self.label_encoder = LabelEncoder()
        chords_list = ['a', 'am', 'bm', 'c', 'd', 'dm', 'e', 'em', 'f', 'g']
        chords_array = np.array(chords_list)
        self.y = self.label_encoder.fit_transform(chords_array)
        # PRINT THE CHORD OPTIONS
        print(self.label_encoder.classes_)
        # ADD A BUFFER TO STORE THE PREDICTIONS AND DEFINE A CONSUME FUNCTION
        self.buffer = deque()
    
    def start_recording(self):
        self.frames.clear()
        print("Listening for threshold level...")
        while True:
            data = np.frombuffer(self.stream.read(self.chunk), dtype=np.int16)
            if np.max(data) > self.threshold:
                print("Threshold level reached. Recording started...")
                return self.record()

    def record(self):
        start_time = time.time()
        while time.time() - start_time < self.duration:
            data = self.stream.read(self.chunk)
            self.lock.acquire()
            self.frames.append(data)
            self.lock.release()
        return self.stop_recording()

    def stop_recording(self):
        print("Recording stopped...")
        with wave.open(self.filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.pa.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.rate)
            self.lock.acquire()
            wf.writeframes(b''.join(self.frames))
            self.lock.release()
        print(f"Recording saved as {self.filename}")
        return self.predict(self.filename)

    def predict(self, filename):
        # LOAD THE AUDIO FILE
        audio_file = filename  # Use the recorded WAV file
        audio_sample, sr = librosa.load(audio_file)
        # PREPROCESS THE FEATURES OF THE AUDIO FILE
        features = generate_features(audio_sample)
        # RESHAPE THE ARRAY TO MATCH THE EXPECTED INPUT SIZE OF THE MODEL
        features = np.expand_dims(features, axis=0)
        # MAKE PREDICTION
        predictions = self.model.predict(features)
        # GET THE PREDICTED CLASS
        predicted_class = np.argmax(predictions)
        # TRANSFORM IT INTO THE CHORD NAME
        predicted_categorical_label = self.label_encoder.inverse_transform([predicted_class])[0]

        # predicted_categorical_label HAS THE PREDICTED CHORD TYPE <- DISPLAY THIS ON THE PAGE
    
        # PRINT THE CHORD NAME
        print(f"For audio file {audio_file}, predicted class: {predicted_categorical_label}")
        # ALSO APPEND TO PREDICTION BUFFER
        self.buffer.append(f"predicted class: {predicted_categorical_label}")

        return predicted_categorical_label

    def consume(self):
        while len(self.buffer) == 0:
            time.sleep(0.2)
        pred = self.buffer[0]
        self.buffer.popleft()
        return pred
# PADS THE FEATURES IF THEY ARE SMALLER THAN THE EXPECTED SHAPE
def padding(array, xx, yy):
    """
    :param array: numpy array
    :param xx: desired height
    :param yy: desirex width
    :return: padded array
    """
    h = array.shape[0]
    w = array.shape[1]
    a = max((xx - h) // 2,0)
    aa = max(0,xx - a - h)
    b = max(0,(yy - w) // 2)
    bb = max(yy - b - w,0)
    return np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant')

# FUNCTION TO NORMALIZE A FEATURE
def normalize(feature):
    return (feature - np.min(feature)) / (np.max(feature) - np.min(feature))

# EXTRACTS THE NEEDED FEATURES FROM AN AUDIO FILE
def generate_features(audio_sample):
    max_size = 1000
    n_mfcc = 13
    sr = 22050

    # EXTRACT STFT, MFCCS, SPECTRAL CENTROID, CHROMA STFT, AND SPECTRAL BANDWIDTH
    stft = librosa.stft(y=audio_sample, n_fft=255, hop_length=512)
    stft = stft[:, :max_size]  # Truncate stft to max_size
    stft = padding(np.abs(stft), 128, max_size)

    mfccs = librosa.feature.mfcc(y=audio_sample, sr=sr, n_mfcc=n_mfcc)
    mfccs = mfccs[:, :max_size]  # Truncate mfccs to max_size
    mfccs = padding(mfccs, 128, max_size)

    spec_centroid = librosa.feature.spectral_centroid(y=audio_sample, sr=sr)
    chroma_stft = librosa.feature.chroma_stft(y=audio_sample, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=audio_sample, sr=sr)

    # TRUNCATE THE SIZE OF A FEATURE IF IT EXCEEDS MAX_SIZE (NEEDED FOR COMPUTATIONAL EFFICIENCY)
    spec_bw_truncated = spec_bw[:, :max_size]
    spec_centroid_truncated = spec_centroid[:, :max_size]
    chroma_stft_truncated = chroma_stft[:, :max_size]

    # CREATE THE IMAGE STACK
    image = np.array([padding(normalize(spec_bw_truncated), 1, max_size)]).reshape(1, max_size)
    image = np.append(image, padding(normalize(spec_centroid_truncated), 1, max_size), axis=0)

    # THESE THREE FEATURES ARE BEING STACKED INTO ONE DIMENSION
    for i in range(0, 9):
        image = np.append(image, padding(normalize(spec_bw_truncated), 1, max_size), axis=0)
        image = np.append(image, padding(normalize(spec_centroid_truncated), 1, max_size), axis=0)
        image = np.append(image, padding(normalize(chroma_stft_truncated), 12, max_size), axis=0)

    # STACK STFT AND MFCCS INTO THEIR OWN DIMENSION
    image = np.dstack((image, stft))
    image = np.dstack((image, mfccs))

    return image