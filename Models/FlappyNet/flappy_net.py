import torch
from torch import nn
import torch.nn.functional as F

class FlappyNet(nn.Module):
    def __init__(self):
        super(FlappyNet, self).__init__()

        self.fc1 = nn.Linear(2, 6)
        self.fc2 = nn.Linear(6, 2)

        self.fitness = 0

        for param in self.parameters():
            nn.init.uniform_(param.data, -1, 1)
            param.requires_grad = False

    def forward(self, x):
        out = self.fc1(x)
        out = torch.sigmoid(out)
        out = self.fc2(out)

        return out

    def move(self, directions):
        input = torch.tensor(directions).float()
        output = self.forward(input)

        if output[0] > output[1]:
            return True
        return False