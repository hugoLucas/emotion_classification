from utils.config_utils import get_config_from_json
from torch import from_numpy, FloatTensor, topk
from torch.autograd.variable import Variable
from data import process_audio, pad_array
from torch.nn.functional import softmax
from utils.model_utils import emotions
from librosa import load as lib_load
from model import BidirectionalLSTM
from torch import load as py_load
from os import path

model_configs, _ = get_config_from_json(path.join('./configs', "config_1.json"))
model = BidirectionalLSTM(model_configs=model_configs)
model.load_state_dict(py_load("/home/hugolucas/PycharmProjects/sound/models/good_model.pt"))
model.eval()

test_file, sr = lib_load("/home/hugolucas/emotion_dataset/test/03-01-08-01-01-02-01.wav", duration=5)
test_file = process_audio(test_file, sr)
test_file = pad_array(test_file)

x = Variable(from_numpy(test_file))
x = x.type(FloatTensor)
x = x.unsqueeze(0)

output = softmax(model(x))
print(output)
_, output = topk(output, k=1)
print(emotions.get(int(output.data[0]), "UNKNOWN"))
