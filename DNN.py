import os
import matplotlib.pyplot as plt
import torch
from torch import nn
from data_load import y_dimension, X_dimension
from train_test import train, test, device, loss_value_plot


class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(X_dimension, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, y_dimension)
        )

    def forward(self, X):
        X = self.flatten(X)
        logits = self.network(X)
        return logits


DNN_model = DNN()
DNN_model.to(device=device)

# 超参数
epochs = 5
lr = 1e-3
momentum = 0.9
optimizer = torch.optim.SGD(DNN_model.parameters(), lr=lr, momentum=momentum)
loss_fn = nn.CrossEntropyLoss()

if os.path.exists('./model/DNN_model.pth'):
    DNN_model.load_state_dict(torch.load('./model/DNN_model.pth'))
else:
    losses, iter = train(DNN_model, optimizer, loss_fn, epochs)
    torch.save(DNN_model.state_dict(), './model/DNN_model.pth')

    loss_value_plot(losses, iter)
    plt.savefig('./loss/DNN_loss.png')

test(DNN_model)
