from numpy import concatenate, swapaxes, zeros
from librosa.feature import mfcc, delta
from torch.utils.data import Dataset
from librosa.core import load
from os import listdir, path


class AudioData(Dataset):
    def __init__(self, configs, training_data=True):
        self.configs = configs
        self.training_data = training_data
        self.file_list = self.load_files()
        self.max_samples = None

    def __getitem__(self, index):
        file_path = self.file_list[index]
        audio_data, sr = load(file_path, duration=self.configs.audio_max_length)

        # Extract audio features
        audio_data = self.process_audio(audio_data, sr)
        if audio_data.shape[0] < 51:
            audio_data = concatenate((audio_data, zeros((51 - audio_data.shape[0], 36))))

        # Extract label out, subtract one to make first label 0 rather than 1
        file_name = self.file_list[index].split()[-1]
        label = int(file_name.split('-')[self.configs.emotion_index]) - 1

        return audio_data, label

    def __len__(self):
        return len(self.file_list)

    def load_files(self):
        """
        Iterates through all folders in train_data directory and stores each folder's contents in a list. Used to make
        indexing of train_data set easier.

        :return: a list containing the file path to each audio file in the train_data set
        :rtype: List[str]
        """
        file_list = []

        repo = self.configs.data_dir if self.training_data else self.configs.test_dir
        folders = filter(lambda f: self.configs.folder_tag in f and '.' not in f, listdir(repo))
        for fld in folders:
            fld_path = path.join(self.configs.data_dir, fld)
            files = filter(lambda f: self.configs.file_type in f, listdir(fld_path))

            for fl in files:
                file_list.append(path.join(fld_path, fl))

        return file_list

    @staticmethod
    def process_audio(audio_data, sr):
        """
        Computes the Mel-Frequency Cepstral Coefficients and their first and second order derivatives. Concatenates then
        all into a single numpy array and the swaps the axis from [n_mfcc, n_samples] to [n_samples, n_mfcc].

        :param audio_data: floating point time series of an audio file
        :param sr: the sample rate at which train_data was loaded
        :return: a feature array of dimension [n_samples, n_mfcc] containing the computed MFCCs and their time
                 derivatives
        """
        mel_freq_coeff = mfcc(y=audio_data, sr=sr, n_mfcc=13, hop_length=int(.10*sr), n_fft=int(.20*sr))
        mel_freq_coeff = mel_freq_coeff[1:, :]

        mel_freq_coeff_delta = delta(mel_freq_coeff)
        mel_freq_coeff_delta_delta = delta(mel_freq_coeff, order=2)

        features = concatenate((mel_freq_coeff, mel_freq_coeff_delta, mel_freq_coeff_delta_delta), axis=0)
        features = swapaxes(features, 0, 1)
        return features

