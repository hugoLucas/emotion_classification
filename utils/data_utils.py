from os import path, listdir, makedirs, rename, rmdir
from utils.config_utils import get_config_from_json
from random import random

"""
This script should be used to create a training and test set from the extracted folder of the 
Audio_Speech_Actors_01-24.zip file from The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) 
web page. As of April 29, 2018 this file can be found at: https://zenodo.org/record/1188976
"""


def create_directories(configs):
    """
    Creates a directory for the training and test sets if they do not already exist.

    :param configs:     a Bunch object of the run's configuration file
    :return:            the paths to the training and test directories
    """
    train_directory, test_directory = configs.data_dir, configs.test_dir
    if not path.isdir(train_directory):
        makedirs(train_directory)
    if not path.isdir(test_directory):
        makedirs(test_directory)

    return train_directory, test_directory


def move_file(folder_path, destination_path, file_name):
    """
    Moves an audio file from its directory to either the train or test set directory.

    :param folder_path:         the path to the directory containing the audio file
    :param destination_path:    the path to the test or train set directory
    :param file_name:           the name of the audio file
    :return:                    None
    """
    rename(path.join(folder_path, file_name), path.join(destination_path, file_name))


def process_folders(configs):
    """
    Iterates through a directory containing folders of audio data and segments them into a training and test set.

    :param configs:     a Bunch object of the run's configuration file
    :return:            None
    """
    train_directory, test_directory = create_directories(configs)
    for folder in filter(lambda v: "." not in v, listdir(configs.raw_data_dir)):
        folder_path = path.join(configs.raw_data_dir, folder)
        for file in filter(lambda v: ".wav" in v, listdir(folder_path)):
            destination = train_directory if random() <= configs.p_train else test_directory
            move_file(folder_path, destination, file)
        rmdir(folder_path)

RUN_CONFIG_FILE = "config_1.json"
model_configs, _ = get_config_from_json(path.join('/home/hugolucas/PycharmProjects/sound/configs', RUN_CONFIG_FILE))

process_folders(model_configs)