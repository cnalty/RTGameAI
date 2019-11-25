import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.norm3 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(4, 4)


        self.fc1 = nn.Linear(10000, 1000)
        self.fc2 = nn.Linear(1000, 4)

        for param in self.parameters():
            param.requires_grad = False
            nn.init.uniform_(param.data, -5, 5)

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
        out = self.pool2(out)


        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)

        return out

    def choose_move(self, image, body, apple):
        image = np.array(image)
        pil_im = Image.fromarray(image)
        transform = transforms.Compose([
            transforms.Resize(200),
        ])
        im = transform(pil_im)
        pil_im.save("test.jpg", "JPEG")
        im = np.transpose(im)
        im = torch.tensor([im]).float()

        output = self.forward(im)
        moves = output.tolist()[0]
        #print(moves)
        max_i = 0
        for i in range(len(moves)):
            if moves[i] > max_i:
                max_i = i
        for i in range(len(moves)):
            if i == max_i:
                moves[i] = True
            else:
                moves[i] = False

        moves.append(False)
        return moves

    def fitness_model(self, fitness_params):
        score = fitness_params[0]
        closeness = fitness_params[1]
        turns = fitness_params[2]
        away = fitness_params[3]

        fitness = score ** 3
        fitness += turns
        fitness -= away * 0.75

        return fitness