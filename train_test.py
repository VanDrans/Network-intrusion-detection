import matplotlib.pyplot as plt
import torch
from torch import nn

from data_load import train_dataloader, test_dataloader, X_dimension, y, y_dimension, train_data, test_data

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(torch.cuda.is_available())
loss_fn = nn.CrossEntropyLoss()


def train(model, optimizer, loss_fn, epochs):
    losses = []
    iter = 0

    for epoch in range(epochs):
        print(f"epoch {epoch + 1}\n-----------------")
        for i, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device).to(torch.float32), y.to(device).to(torch.float32)
            X = X.reshape(X.shape[0], 1, X_dimension)
            y_pred = model(X)
            loss = loss_fn(y_pred, y.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"loss: {loss.item()}\t[{(i + 1) * len(X)}/{len(train_data)}]")
                iter += 1
                losses.append(loss.item())

    return losses, iter


def test(model):
    positive = 0
    negative = 0
    with torch.no_grad():
        iter = 0
        loss_sum = 0
        for X, y in test_dataloader:
            X, y = X.to(device).to(torch.float32), y.to(device).to(torch.float32)
            X = X.reshape(X.shape[0], 1, X_dimension)
            y_pred = model(X)
            loss = loss_fn(y_pred, y.long())
            loss_sum += loss.item()
            iter += 1
            for item in zip(y_pred, y):
                if torch.argmax(item[0]) == item[1]:
                    positive += 1
                else:
                    negative += 1
    acc = positive / (positive + negative)
    avg_loss = loss_sum / iter
    print("Accuracy:", acc)
    print("Average Loss:", avg_loss)


def loss_value_plot(losses, iter):
    plt.figure()
    plt.plot([i for i in range(1, iter + 1)], losses)
    plt.xlabel('Iterations (×100)')
    plt.ylabel('Loss Value')
