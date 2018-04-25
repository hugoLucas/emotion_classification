from utils.config_utils import get_config_from_json
from torch import from_numpy, FloatTensor, topk
from torch.autograd.variable import Variable
from data import process_audio, pad_array
from torch.nn.functional import softmax
from librosa import load as lib_load
from model import BidirectionalLSTM
from torch import load as py_load
from os import path

emotions = {
    0: "Neutral",
    1: "Calm",
    2: "Happy",
    3: "Sad",
    4: "Angry",
    5: "Fearful",
    6: "Disgust",
    7: "Surprised",
}


def get_emotion(output_index):
    return emotions.get(output_index, "ERROR")


def load_data_from_path(data_path, audio_duration):
    test_file, sr = lib_load(data_path, duration=audio_duration)
    test_file = process_audio(test_file, sr)
    test_file = pad_array(test_file)

    return test_file


def to_variable(audio_data):
    x = Variable(from_numpy(audio_data))
    x = x.type(FloatTensor)
    x = x.unsqueeze(0)

    return x


class Predictor:
    def __init__(self, config_path, model_state):
        self.configs, _ = get_config_from_json(config_path)
        self.model = BidirectionalLSTM(model_configs=self.configs)
        self.model.load_state_dict(py_load(model_state))
        self.model.eval()

    def make_single_prediction(self, audio_data):
        output = self.model(audio_data)
        output = softmax(output)
        _, output = topk(output, k=1)

        return get_emotion(output.data[0])
