import os
import matplotlib.pyplot as plt
import torch
from torch import nn
from data_load import y_dimension, X_dimension
from train_test import train, test, device, loss_value_plot


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 3, kernel_size=2),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(3, 8, kernel_size=2),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(8, 16, kernel_size=2)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(432, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, y_dimension)
        )

    def forward(self, X):
        X = self.backbone(X)
        X = self.flatten(X)
        logits = self.fc(X)
        return logits


CNN_model = CNN()
CNN_model.to(device=device)

epochs = 1
lr = 1e-3
momentum = 0.9
optimizer = torch.optim.SGD(CNN_model.parameters(), lr=lr, momentum=momentum)
loss_fn = nn.CrossEntropyLoss()

if os.path.exists('./model/CNN_model.pth'):
    CNN_model.load_state_dict(torch.load('./model/CNN_model.pth'))
else:
    losses, iter = train(CNN_model, optimizer, loss_fn, epochs)
    torch.save(CNN_model.state_dict(), './model/CNN_model.pth')

    loss_value_plot(losses, iter)
    plt.savefig('./loss/CNN_loss.png')

test(CNN_model)
