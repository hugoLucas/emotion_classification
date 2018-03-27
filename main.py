from utils.config_utils import get_config_from_json
from torch.autograd.variable import Variable
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from model import BidirectionalLSTM
from torch.optim import Adam
from data import AudioData
from os.path import join


# Any parameters that may change from run-to-run
RUN_CONFIG_FILE = "config_1.json"

# Run Configs
model_configs, _ = get_config_from_json(join('./configs', RUN_CONFIG_FILE))

# Data
audio_data = AudioData(configs=model_configs)
train_loader = DataLoader(dataset=audio_data, batch_size=model_configs.batch_size, shuffle=True, num_workers=1)

# Model
audio_model = BidirectionalLSTM(model_configs=model_configs)
audio_model.cuda()

# Training Params
loss_fn = CrossEntropyLoss()
optimizer = Adam(audio_model.parameters(), lr=model_configs.learning_rate)

running_loss = 0.0
for i, data in enumerate(train_loader, 0):
    # Get the inputs and wrap them
    inputs, labels = data
    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

    # Zero out the gradients
    optimizer.zero_grad()

    # Propagate forward, backwards
    outputs = audio_model(inputs)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()

    # Log results
    running_loss += loss.data[0]
    print(running_loss)
    break
