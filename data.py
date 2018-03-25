from torch.utils.data import Dataset
from os import listdir, path


class AudioData(Dataset):
    def __init__(self, configs):
        self.configs = configs

        self.file_list = self.load_files()

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

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
            file_list.extend(path.join(fld_path, files))

        return file_list
