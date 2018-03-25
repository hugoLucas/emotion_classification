from utils.config_utils import get_config_from_json
from data import AudioData
from os.path import join

RUN_CONFIG_FILE = "config_1.json"

model_configs = get_config_from_json(join('./configs', RUN_CONFIG_FILE))
data_set = AudioData(config=model_configs)
