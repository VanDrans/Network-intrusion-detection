import os
import matplotlib.pyplot as plt
import torch
from torch import nn
from data_load import y_dimension, X_dimension
from train_test import train, test, device, loss_value_plot, loss_fn


class CNN_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=2),
            nn.MaxPool1d(2, 2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=2),
            nn.MaxPool1d(2, 2)
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=128,
                            num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, y_dimension)

    def forward(self, x):
        x = self.backbone(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)

        return x


CNN_LSTM_model = CNN_LSTM()
CNN_LSTM_model.to(device=device)

epochs = 15
lr = 0.001
momentum = 0.9
optimizer = torch.optim.SGD(CNN_LSTM_model.parameters(), lr=lr, momentum=0.9)

if os.path.exists('model/CNN_LSTM_model.pth'):
    CNN_LSTM_model.load_state_dict(torch.load('model/CNN_LSTM_model.pth'))
else:
    losses, iter = train(CNN_LSTM_model, optimizer, loss_fn, epochs)
    torch.save(CNN_LSTM_model.state_dict(), 'model/CNN_LSTM_model.pth')

    loss_value_plot(losses, iter)
    plt.savefig('./loss/CNN_LSTM_loss.png')

test(CNN_LSTM_model)
