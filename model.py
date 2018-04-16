from torch.nn import Module, LSTM, Linear
from torch.nn.functional import relu
from torch import cat


class BidirectionalLSTM(Module):
    """
    Model is based on model found in paper: Emotion Recognition From Speech With Recurrent Neural Networks.
    My GPU cannot handle the full model however so I have cut down on some of the parameters.
    """

    def __init__(self, model_configs):
        super(BidirectionalLSTM, self).__init__()

        self.configs = model_configs
        self.n_features = self.configs.audio_sample_rate * self.configs.audio_max_length
        self.lstm_1 = LSTM(input_size=1, hidden_size=self.configs.lstm_output_dim,
                           num_layers=self.configs.lstm_layers, batch_first=True, dropout=1, bidirectional=True)
        self.dense_2 = Linear(in_features=2 * self.configs.lstm_output_dim,
                              out_features=self.configs.dense_1_output_dim)
        self.dense_3 = Linear(in_features=self.configs.dense_1_output_dim, out_features=self.configs.dense_2_output_dim)
        self.dense_4 = Linear(in_features=self.configs.dense_2_output_dim, out_features=self.configs.dense_3_output_dim)

    def forward(self, x):
        x, _ = self.lstm_1(x)
        x = cat((x[:, -1, self.configs.lstm_output_dim:], x[:, 0, :self.configs.lstm_output_dim]), 1)
        x = relu(self.dense_2(x))
        x = relu(self.dense_3(x))
        x = self.dense_4(x)
        return x


class SimpleLSTM(Module):
    def __init__(self, model_configs):
        super(SimpleLSTM, self).__init__()

        self.configs = model_configs
        self.lstm_1 = LSTM(input_size=1, hidden_size=self.configs.lstm_output_dim, num_layers=self.configs.lstm_layers,
                           batch_first=True)
        self.dense_2 = Linear(in_features=self.configs.lstm_output_dim, out_features=self.configs.dense_1_output_dim)
        self.dense_3 = Linear(in_features=self.configs.dense_1_output_dim, out_features=self.configs.dense_2_output_dim)
        self.dense_4 = Linear(in_features=self.configs.dense_2_output_dim, out_features=self.configs.dense_3_output_dim)

    def forward(self, x):
        x, _ = self.lstm_1(x)
        x = relu(self.dense_2(x[-1]))
        x = relu(self.dense_3(x))
        x = self.dense_4(x)
        return x