from numpy import concatenate, swapaxes, zeros
from librosa.feature import mfcc, delta
from torch.utils.data import Dataset
from librosa.core import load
from os import listdir, path


def process_audio(audio_data, sr):
    """
    Computes the Mel-Frequency Cepstral Coefficients and their first and second order derivatives. Concatenates then
    all into a single numpy array and the swaps the axis from [n_mfcc, n_samples] to [n_samples, n_mfcc].

    :param audio_data: floating point time series of an audio file
    :param sr: the sample rate at which train_data was loaded
    :return: a feature array of dimension [n_samples, n_mfcc] containing the computed MFCCs and their time
             derivatives
    """
    mel_freq_coeff = mfcc(y=audio_data, sr=sr, n_mfcc=13, hop_length=int(.10 * sr), n_fft=int(.20 * sr))
    mel_freq_coeff = mel_freq_coeff[1:, :]

    mel_freq_coeff_delta = delta(mel_freq_coeff, width=7)
    mel_freq_coeff_delta_delta = delta(mel_freq_coeff, width=7, order=2)

    features = concatenate((mel_freq_coeff, mel_freq_coeff_delta, mel_freq_coeff_delta_delta), axis=0)
    features = swapaxes(features, 0, 1)
    return features


def pad_array(audio_data):
    """
    Pads an array with zeros if the audio file used to generate the array is less than a certain value.

    :param audio_data:  a [1, N] dimensional array created by the Librosa.core.load method
    :return: the input array padded with 0's if necessary
    """
    if audio_data.shape[0] < 51:
        audio_data = concatenate((audio_data, zeros((51 - audio_data.shape[0], 36))))
    return audio_data


class AudioData(Dataset):

    """
    Implementation of the pytorch.utils.data.Dataset class. Loads and processes the RAVDESS dataset for easier training.
    """

    def __init__(self, configs, training_data=True):
        """
        :param configs:         Bunch object of this run's configuration file
        :param training_data:   if True this object will load in data to be used for training a model, if False the
                                object will load data for testing a model
        """
        self.configs = configs
        self.training_data = training_data
        self.file_list, self.dir = self.load_files()
        self.max_samples = None

    def __getitem__(self, index):
        """
        Takes a random file path from file_list and loads in the corresponding data file. Extracts MFCC features from
        the audio data as well as the label number from the file name. Pads the audio data if required.

        :param index:   a random index in file_list
        :return:        a Numpy array containing MFCC features of the audio file as well as the correct label for said
                        file
        """
        file_name = self.file_list[index]
        audio_data, sr = load(path.join(self.dir, file_name), duration=self.configs.audio_max_length)

        audio_data = process_audio(audio_data, sr)
        audio_data = pad_array(audio_data)

        file_name = self.file_list[index].split()[-1]

        # Subtract one as file labels range from 1 to 8 while Softmax function only takes in 0 to 7
        label = int(file_name.split('-')[self.configs.emotion_index]) - 1

        return audio_data, label

    def load_files(self):
        """
        Iterates through all files in train_data directory and stores each folder's contents in a list. Used to make
        indexing of train_data set easier.

        :return: a list containing the file path to each audio file in the train_data set
        :rtype: List[str]
        """

        repo = self.configs.data_dir if self.training_data else self.configs.test_dir
        file_list = list(filter(lambda f: self.configs.file_type in f, listdir(repo)))

        return file_list, repo

    def __len__(self):
        return len(self.file_list)