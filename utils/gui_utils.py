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
    """
    Creates a thread in order to pass a recorded audio file to Predictor to avoid blocking the main UI thread.
    """
    def __init__(self, predictor, file_path, callback_fn):
        """
        :param predictor:       a Predictor object
        :param file_path:       the file path to the User's recording
        :param callback_fn:     a function to call once the prediction is made
        """
        threading.Thread.__init__(self)
        self.cb = callback_fn
        self.path = file_path
        self.pred = predictor

    def run(self):
        """
        Loads recording and passes it to the model to make a prediction.
        :return: None
        """
        data = load_data_from_path(self.path)
        data = to_variable(data)

        emotion = self.pred.make_single_prediction(audio_data=data)
        self.cb(emotion)


class RecorderThread(threading.Thread):
    """
    Creates a thread to record the User's voice to avoid blocking the main UI thread.
    """
    def __init__(self, callback_fn):
        """
        :param callback_fn:     a function to call once the recording is complete
        """
        threading.Thread.__init__(self)
        self.cb = callback_fn

    def run(self):
        """
        Records a User's voice in order to classify at a later time.
        :return:
        """
        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT, channels=CHANNEL, rate=RATE, input=True, frames_per_buffer=CHUNK)
        print("*** recording ***")

        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        print("*** done ***")

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
