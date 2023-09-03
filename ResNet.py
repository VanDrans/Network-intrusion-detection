import os
import matplotlib.pyplot as plt
import torch
from torch import nn
from data_load import y_dimension, X_dimension
from train_test import train, test, device,loss_value_plot


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = nn.Sequential(
            ResNetBlock(64, 64),
            ResNetBlock(64, 64)
        )

        self.layer2 = nn.Sequential(
            ResNetBlock(64, 128, stride=2),
            ResNetBlock(128, 128)
        )

        self.layer3 = nn.Sequential(
            ResNetBlock(128, 256, stride=2),
            ResNetBlock(256, 256)
        )

        self.layer4 = nn.Sequential(
            ResNetBlock(256, 512, stride=2),
            ResNetBlock(512, 512)
        )

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, y_dimension)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


ResNet_model = ResNet()
ResNet_model.to(device=device)

epochs = 1
lr = 0.001
momentum = 0.9
optimizer = torch.optim.SGD(ResNet_model.parameters(), lr=lr, momentum=momentum)
loss_fn = nn.CrossEntropyLoss()

if os.path.exists('./model/ResNet_model.pth'):
    ResNet_model.load_state_dict(torch.load('./model/ResNet_model.pth'))
else:
    losses, iter = train(ResNet_model, optimizer, loss_fn, epochs)
    torch.save(ResNet_model.state_dict(), './model/ResNet_model.pth')

    loss_value_plot(losses, iter)
    plt.savefig('./loss/ResNet_loss.png')

test(ResNet_model)
