import numpy as np
import matplotlib.pyplot as plt
import torch
from rnn import scaler, train_data_normalized, train_window, \
    LSTM
from config import test_data_size, days_to_predict, df

test_inputs = train_data_normalized[-train_window:].tolist()

model = LSTM()
model.load_state_dict(torch.load('finished_models/lastmodel.pt', map_location=torch.device('cpu')))

model.eval()

for i in range(test_data_size + days_to_predict):
    seq = torch.FloatTensor(test_inputs[-train_window:])
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        test_inputs.append(model(seq).item())

actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:] ).reshape(-1, 1))
x = np.arange(len(df)-test_data_size, len(df)+days_to_predict, 1)
actual_predictions[0] = df["VALUE, USD"].iloc[len(df)-test_data_size]

plt.title('Актуальні дані/Передбачення')
plt.ylabel('ВВП за квартал')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(df['VALUE, USD'])
plt.plot(x, actual_predictions)
plt.show()

with open('predictions.txt', 'w') as f:
    for elem in actual_predictions[test_data_size:]:
        f.write("{:.2f}\n".format(float(elem.item())))