import torch
import torch.nn as nn
import torch.nn.functional as F

class LookModel(nn.Module):
    def __init__(self):
        super(LookModel, self).__init__()

        # Looks in 8 directions, first 8 nodes distance to object, second 8 is whether object is apple or wall
        self.fc1 = nn.Linear(16, 12)
        nn.init.xavier_normal_(self.fc1.weight.data)
        self.fc2 = nn.Linear(12, 12)
        nn.init.xavier_normal_(self.fc2.weight.data)
        self.fc3 = nn.Linear(12, 4) # 4 directions for output layer
        nn.init.xavier_normal_(self.fc3.weight.data)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)

        return out
