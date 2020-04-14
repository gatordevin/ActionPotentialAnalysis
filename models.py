import torch.nn as nn
import torch.nn.functional as F

class SingleLSTMLayer(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq.view(len(input_seq) ,1, -1))
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

class FullyConnectedModel(nn.Module):
    def __init__(self, input_size=400, hidden_size=300, num_of_hidden_layers=2, output_size=2):
        super().__init__()
        self.num_of_hidden_layers = num_of_hidden_layers
        self.l1 = nn.Linear(input_size, hidden_size).cuda()
        self.l2 = nn.Linear(hidden_size, hidden_size).cuda()
        self.relu = nn.ReLU().cuda()
        self.l3 = nn.Linear(hidden_size, output_size).cuda()

    def forward(self, input_seq):
        x = input_seq
        x = self.l1(x)
        x = self.relu(x)
        for hidden_layer in range(self.num_of_hidden_layers):
            x = self.l2(x)
            x = self.relu(x)
        x = self.l3(x)
        return F.log_softmax(x)

class LSTMDeepModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size).cuda()
        self.dropout = nn.Dropout2d(0.5).cuda()
        self.dense = nn.Linear(hidden_layer_size, hidden_layer_size).cuda()
        self.relu = nn.ReLU().cuda()
        self.out = nn.Linear(hidden_layer_size, output_size).cuda()

    def forward(self, input_seq):
        x, _ = self.lstm(input_seq.view(len(input_seq) ,1, -1))
        x = self.dropout(x)
        x = self.relu(self.dense(x))
        x = self.out(x.view(len(input_seq), -1))
        return x[-1]