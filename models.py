## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I
from torch.autograd import Variable


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.MaxPool2d(2, 2), nn.ReLU(),
            nn.BatchNorm2d(32))
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3), nn.MaxPool2d(2, 2), nn.ReLU(),
            nn.BatchNorm2d(32))

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3), nn.MaxPool2d(2, 2), nn.ReLU(),
            nn.BatchNorm2d(64))

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3), nn.MaxPool2d(2, 2), nn.ReLU(),
            nn.BatchNorm2d(64))

        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, 3), nn.MaxPool2d(2, 2), nn.ReLU(),
            nn.BatchNorm2d(128))

        self.fc1 = nn.Linear(128 * 5 * 5, 1024)
        self.fc1_drop = nn.Dropout(p=0.4)
        # self.fc2 = nn.Linear(1024, 1024)
        # self.fc2_drop = nn.Dropout(p=0.4)
        self.fc3 = nn.Linear(1024, 136)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)

        # x = F.relu(self.fc2(x))
        # x = self.fc2_drop(x)

        x = self.fc3(x)

        return x
