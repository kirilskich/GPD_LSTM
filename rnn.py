import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from config import test_data_size, train_window, df



data = df.set_index('DATE').to_numpy().astype(float)
data = data.flatten()

train_data = data[:-test_data_size]
test_data = data[-test_data_size:]

scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(train_data.reshape(-1, 1))
train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq

train_inout_seq = create_inout_sequences(train_data_normalized, train_window)


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1, num_layers=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, bidirectional=True)
        #self.fc1 = nn.Linear(hidden_layer_size, 256)
        #self.fc2 = nn.Linear(256, 64)
        #self.fc3 = nn.Linear(64, 1)
        self.linear = nn.Linear(hidden_layer_size*2, output_size)

    def init_hidden_state(self, h_l):
        self.hidden_cell = (torch.zeros(self.num_layers*2,1,h_l),
                            torch.zeros(self.num_layers*2,1,h_l))

    def forward(self, input_seq):
        self.init_hidden_state(self.hidden_layer_size)
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        prediction = self.linear(lstm_out.view(len(lstm_out), -1))
        #prediction = F.relu(self.fc1(lstm_out.view(len(lstm_out), -1)))
        #prediction = F.relu(self.fc2(prediction))
        #prediction = self.fc3(prediction)

        return prediction[-1]





