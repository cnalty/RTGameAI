mport torch
from torch import nn
import torch.nn.functional as F

class CharNet(nn.Module):
    def __init__(self, out_chars):
        super(CharNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.norm3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.norm4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(3136, 3136)
        self.fc2 = nn.Linear(3136, out_chars)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.pool1(out)
        out = self.conv3(out)
        out = self.norm3(out)
        out = F.relu(out)
        out = self.conv4(out)
        out = self.norm4(out)
        out = F.relu(out)
        out = self.pool2(out)

        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)

        return out