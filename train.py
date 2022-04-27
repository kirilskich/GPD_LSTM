import torch
from rnn import LSTM, nn, train_inout_seq
from config import epochs

model = LSTM()
loss_function = nn.L1Loss() #MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

min = 1


for i in range(epochs):
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if single_loss.item() < min:
        min = single_loss.item()
        torch.save(model.state_dict(), 'models/bestmodel.pt')
        print(f'EPOCH: {i+1} | BEST LOSS: {single_loss.item()}')

    print(f'EPOCH: {i + 1} | LOSS: {single_loss.item()}')
torch.save(model.state_dict(), 'models/lastmodel.pt')
