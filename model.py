from torch.nn import Module, LSTM, Linear, ReLU, Softmax


class BidirectionalLSTM(Module):

    def __init__(self, model_configs):
        super(BidirectionalLSTM, self).__init__()

        self.configs = model_configs
        self.lstm_1 = LSTM(input_size=1, hidden_size=self.configs.lstm_1_output_dim,
                           num_layers=self.configs.lstm_layers, batch_first=True, dropout=1, bidirectional=True)
        self.dense_2 = Linear(in_features=self.configs.lstm_1_output_dim, out_features=self.configs.dense_1_output_dim)
        self.dense_3 = Linear(in_features=self.configs.dense_1_output_dim, out_features=self.configs.dense_2_output_dim)

    def forward(self, x):
        h_t, c_t = None, None
        for val in x:
            h_t, c_t = self.lstm_1(val)
        x = ReLU(self.dense_2(h_t))
        x = Softmax(self.dense_3(x))
        return x
