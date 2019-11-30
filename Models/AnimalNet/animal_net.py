import torch
import torch.nn as nn
import torch.nn.functional as F


class AnimalNet(nn.Module):
    def __init__(self):
        super(AnimalNet, self).__init__()

        self.fc1 = nn.Linear(26, 18)
        self.fc2 = nn.Linear(18, 2)

        self.fitness = 0

        for param in self.parameters():
            nn.init.uniform_(param.data, -1, 1)
            param.requires_grad = False

    def forward(self, x):
        out = self.fc1(x)
        out = F.sigmoid(out)
        out = self.fc2(out)

        return out

    def steer(self, directions):
        dir = torch.tensor(directions).float()
        outputs = self.forward(dir)

        if outputs[0] > outputs[1]:
            return "L"
        else:
            return "R"