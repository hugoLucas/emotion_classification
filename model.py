from torch.nn import Module, LSTM, Linear, ReLU, Softmax
from torch.autograd.variable import Variable
from torch import randn


class BidirectionalLSTM(Module):

    def __init__(self, model_configs):
        super(BidirectionalLSTM, self).__init__()

        self.configs = model_configs
        self.lstm_1 = LSTM(input_size=1, hidden_size=self.configs.lstm_output_dim,
                           num_layers=self.configs.lstm_layers, batch_first=False, dropout=1, bidirectional=True)
        self.dense_2 = Linear(in_features=2 * self.configs.lstm_output_dim,
                              out_features=self.configs.dense_1_output_dim)
        self.reul = ReLU()
        self.dense_3 = Linear(in_features=self.configs.dense_1_output_dim, out_features=self.configs.dense_2_output_dim)
        self.softmax = Softmax()

    def forward(self, x):
        dim1, dim3 = self.configs.lstm_layers * 2 * self.configs.batch_size, self.configs.lstm_output_dim

        h1, h2 = randn(dim1, 1, dim3), randn(dim1, 1, dim3)
        h1, h2 = Variable(h1.cuda()), Variable(h2.cuda())

        hidden = (h1, h2)
        for val in x[0]:
            out, hidden = self.lstm_1(val.view(1, 1, -1), hidden)
        x = self.reul(self.dense_2(out))
        x = self.softmax(self.dense_3(x))
        print(x)
        return x
