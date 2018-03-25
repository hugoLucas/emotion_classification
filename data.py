from torch.utils.data import Dataset
from librosa.core import load
from os import listdir, path
from numpy import pad


class AudioData(Dataset):
    def __init__(self, configs):
        self.configs = configs
        self.file_list = self.load_files()

    def __getitem__(self, index):
        file_path = self.file_list[index]
        audio_data, _ = load(file_path, sr=self.configs.audio_sample_rate, duration=self.configs.audio_max_length)

        if audio_data.size < self.configs.audio_max_length:
            pad_length = self.configs.audio_max_length - audio_data.size
            pad(audio_data, (0, pad_length), 'constant', constant_values=0)

        return audio_data

    def __len__(self):
        return len(self.file_list)

    def load_files(self):
        """
        Iterates through all folders in data directory and stores each folder's contents in a list. Used to make
        indexing of data set easier.

        :return: a list containing the file path to each audio file in the data set
        :rtype: List[str]
        """
        file_list = []

        folders = filter(lambda f: self.configs.folder_tag in f, listdir(self.configs.data_dir))
        for fld in folders:
            fld_path = path.join(self.configs.data_dir, fld)
            files = filter(lambda f: self.configs.file_type in f, listdir(fld_path))

            for fl in files:
                file_list.append(path.join(fld_path, fl))

        return file_list
