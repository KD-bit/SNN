import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *

steps = STEPS

class CifarNet(nn.Module):
    def __init__(self):
        super(CifarNet, self).__init__()

        self.conv0 = nn.Conv2d(3, 128, 3, 1, 1, bias=None)
        self.bn0 = tdBatchNorm(128)
        self.conv1 = nn.Conv2d(128, 256, 3, 1, 1, bias=None)
        self.bn1 = tdBatchNorm(256)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(256, 512, 3, 1, 1, bias=None)
        self.bn2 = tdBatchNorm(512)
        self.pool2 = nn.AvgPool2d(2)
        self.conv3 = nn.Conv2d(512, 1024, 3, 1, 1, bias=None)
        self.bn3 = tdBatchNorm(1024)
        self.conv4 = nn.Conv2d(1024, 512, 3, 1, 1, bias=None)
        self.bn4 = tdBatchNorm(512)
        self.fc1 = nn.Linear(8 * 8 * 512, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

        self.conv0_s = tdLayer(self.conv0, self.bn0)
        self.conv1_s = tdLayer(self.conv1, self.bn1)
        self.pool1_s = tdLayer(self.pool1)
        self.conv2_s = tdLayer(self.conv2, self.bn2)
        self.pool2_s = tdLayer(self.pool2)
        self.conv3_s = tdLayer(self.conv3, self.bn3)
        self.conv4_s = tdLayer(self.conv4, self.bn4)
        self.fc1_s = tdLayer(self.fc1)
        self.fc2_s = tdLayer(self.fc2)
        self.fc3_s = tdLayer(self.fc3)

        self.spike = LIFSpike()

    def forward(self, x):
        x = self.conv0_s(x)
        x = self.spike(x)
        x = self.conv1_s(x)
        x = self.spike(x)
        x = self.pool1_s(x)
        x = self.spike(x)
        x = self.conv2_s(x)
        x = self.spike(x)
        x = self.pool2_s(x)
        x = self.spike(x)
        x = self.conv3_s(x)
        x = self.spike(x)
        x = self.conv4_s(x)
        x = self.spike(x)
        x = x.view(x.shape[0], -1, x.shape[4])
        x = self.fc1_s(x)
        x = self.spike(x)
        x = self.fc2_s(x)
        x = self.spike(x)
        x = self.fc3_s(x)
        x = self.spike(x)
        out = torch.sum(x, dim=2) / steps  # [N, neurons, steps]
        return out

