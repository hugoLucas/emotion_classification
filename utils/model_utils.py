from utils.config_utils import get_config_from_json
from torch import from_numpy, FloatTensor, max
from torch.autograd.variable import Variable
from data import process_audio, pad_array
from torch.nn.functional import softmax
from librosa import load as lib_load
from model import BidirectionalLSTM
from torch.cuda import is_available
from torch import load as py_load


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


def load_data_from_path(data_path, audio_duration=5):
    audio_data, sr = lib_load(data_path, duration=audio_duration)
    audio_data = process_audio(audio_data, sr)
    audio_data = pad_array(audio_data)

    return audio_data


def to_variable(audio_data):
    x = Variable(from_numpy(audio_data))
    x = x.type(FloatTensor)
    x = x.unsqueeze(0)

    return x


class Predictor:
    def __init__(self, config_path, model_state):
        self.configs, _ = get_config_from_json(config_path)
        self.model = BidirectionalLSTM(model_configs=self.configs)

        if is_available():
            self.model.load_state_dict(py_load(model_state))
        else:
            self.model.load_state_dict(py_load(model_state, map_location=lambda storage, loc: storage))

        self.model.eval()

    def make_single_prediction(self, audio_data):
        output = self.model(audio_data)
        output = softmax(output)
        val, index = max(output, 1)

        if is_available():
            index = index.data[0]
        else:
            index = index.data.cpu().numpy()[0][0]

        return get_emotion(index)
