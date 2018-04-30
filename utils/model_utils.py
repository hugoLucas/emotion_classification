from utils.config_utils import get_config_from_json
from torch import from_numpy, FloatTensor, max
from torch.autograd.variable import Variable
from data import process_audio, pad_array
from torch.nn.functional import softmax
from librosa import load as lib_load
from model import BidirectionalLSTM
from torch.cuda import is_available
from torch import load as py_load


# Label to emotional state map
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
    """
    Given the model's output, method will return string representation of the index.
    :param output_index:    a number from 0 to 7.
    :return:                string for the label emotion
    """
    return emotions.get(output_index, "ERROR")


def load_data_from_path(data_path, audio_duration=5):
    """
    Loads in an audio file given that file's full path.

    :param data_path:           path to .wav file
    :param audio_duration:      the max length to load, five seconds by default
    :return:                    numpy array of the loaded file
    """
    audio_data, sr = lib_load(data_path, duration=audio_duration)
    audio_data = process_audio(audio_data, sr)
    audio_data = pad_array(audio_data)

    return audio_data


def to_variable(audio_data):
    """
    Takes in a numpy array and converts into a PyTorch Variable.

    :param audio_data:  a numpy array
    :return:            Variable object of numpy array
    """
    x = Variable(from_numpy(audio_data))
    x = x.type(FloatTensor)
    x = x.unsqueeze(0)

    return x


class Predictor:
    """
    Class loads in a previously created model's state and allows for the analysis of a single audio sample at a time.
    """
    def __init__(self, config_path, model_state):
        """
        :param config_path: path to the configuration file used to create saved model
        :param model_state: path to .pt file output by Pytorch's save method
        """
        self.configs, _ = get_config_from_json(config_path)
        self.model = BidirectionalLSTM(model_configs=self.configs)

        if is_available():
            self.model.load_state_dict(py_load(model_state))
        else:
            self.model.load_state_dict(py_load(model_state, map_location=lambda storage, loc: storage))

        self.model.eval()

    def make_single_prediction(self, audio_data):
        """
        Given a Variable object, method will pass data to model and output a String representation of the model's
        output.

        :param audio_data:  a Pytorch Variable object
        :return:            string of emotional state detected
        """
        output = self.model(audio_data)
        output = softmax(output)
        val, index = max(output, 1)

        if is_available():
            index = index.data[0]
        else:
            index = index.data.cpu().numpy()[0][0]

        return get_emotion(index)
