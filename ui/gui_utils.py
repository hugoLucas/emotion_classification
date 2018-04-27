import threading
import pyaudio
import wave
from utils.model_utils import load_data_from_path, to_variable

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNEL = 2
RATE = 44100
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "./output.wav"


class PredictionThread(threading.Thread):
    def __init__(self, predictor, file_path, callback_fn):
        threading.Thread.__init__(self)
        self.cb = callback_fn
        self.path = file_path
        self.pred = predictor

    def run(self):
        data = load_data_from_path(self.path)
        data = to_variable(data)

        emotion = self.pred.make_single_prediction(audio_data=data)
        self.cb(emotion)


class RecorderThread(threading.Thread):
    def __init__(self, callback_fn):
        threading.Thread.__init__(self)
        self.cb = callback_fn

    def run(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT, channels=CHANNEL, rate=RATE, input=True, frames_per_buffer=CHUNK)
        print("* recording")

        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        print("* done recording")

        stream.stop_stream()
        stream.close()
        audio.terminate()

        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNEL)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        self.cb()


def center_window(root, width=800, height=400):
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    x = (screen_width / 2) - (width / 2)
    y = (screen_height / 2) - (height / 2)

    root.geometry('%dx%d+%d+%d' % (width, height, x, y))