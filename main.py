from utils.config_utils import get_config_from_json
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from model import BidirectionalLSTM
from trainer import AudioTrainer
from logger import AudioLogger
from torch.optim import Adam
from data import AudioData

# This import must be maintained in order for script to work on paperspace
from os import path

# Any parameters that may change from run-to-run
RUN_CONFIG_FILE = "config_1.json"

# Run Configs
model_configs, _ = get_config_from_json(path.join('./configs', RUN_CONFIG_FILE))

# Training Data
train_data = AudioData(configs=model_configs)
train_loader = DataLoader(dataset=train_data, batch_size=model_configs.batch_size, shuffle=True, num_workers=4)

# Test Data
test_data = AudioData(configs=model_configs, training_data=False)
test_loader = DataLoader(dataset=train_data, batch_size=model_configs.batch_size, shuffle=True, num_workers=4)

# Model
audio_model = BidirectionalLSTM(model_configs=model_configs)
audio_model.cuda()

# Training Params
loss_fn = CrossEntropyLoss()
optimizer = Adam(audio_model.parameters(), lr=model_configs.learning_rate)

# Logger
logger = AudioLogger("./run.json")

# Train Model
trainer = AudioTrainer(configs=model_configs, model=audio_model, train_loader=train_loader, loss_fn=loss_fn,
                       optimizer=optimizer, logger=logger, load_path="./models/mfcc.pt",
                       save_path="./models/mfcc_2.pt", test_loader=test_loader)
trainer.load_model()
trainer.train()
