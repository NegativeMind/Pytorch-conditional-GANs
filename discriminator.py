# -*- coding: utf-8 -*-

import torch
from torch import nn, optim
import torch.nn.functional as F

class Discriminator(nn.Module):
    """ Discriminator """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1  = nn.Linear(64 * 28 * 28 + 1000, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.fc3 = nn.Linear(10, 1000)


    def forward(self, x, labels):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, 28, 28)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = x.view(batch_size, 64 * 28 * 28)
        y_ = self.fc3(labels)
        y_ = F.relu(y_)
        x = torch.cat([x, y_], 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return F.sigmoid(x)