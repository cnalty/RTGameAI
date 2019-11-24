import torch
import torch.nn as nn
import torch.nn.functional as F

class LookModel(nn.Module):
    def __init__(self):
        super(LookModel, self).__init__()

        # Looks in 8 directions dist to object, 2nd 8 if object is apple, 3rd 8 if object is tail
        self.fc1 = nn.Linear(24, 16)
        nn.init.uniform_(self.fc1.weight.data, -5, 5)
        self.fc2 = nn.Linear(16, 16)
        nn.init.uniform_(self.fc2.weight.data, -5, 5)
        self.fc3 = nn.Linear(16, 4) # 4 directions for output layer
        nn.init.uniform_(self.fc3.weight.data, -5, 5)
        self.fitness = None

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)

        return out

    def choose_move(self, image, body, apple):
        head = body[-1]
        walls = []

        walls.append(head[0]) #West
        walls.append(800 - head[0]) #East
        walls.append(head[1]) #North
        walls.append(800 - head[1]) #South
        walls.append(self.hypot(min(walls[0], walls[2]))) #NW
        walls.append(self.hypot(min(walls[1], walls[2]))) #NE
        walls.append(self.hypot(min(walls[0], walls[3]))) #SW
        walls.append(self.hypot(min(walls[1], walls[3]))) #SE

        apples = [0 for _ in range(8)]
        tail = [0 for _ in range(8)]

        # check if apple in vision
        if apple[0] == head[0]:
            if apple[1] < head[1]:
                apples[2] = 1
            else:
                apples[3] = 1
        elif apple[1] == head[1]:
            if apple[0] < head[0]:
                apples[0] = 1
            else:
                apples[1] = 1
        elif abs(apple[0] - head[0]) == abs(apple[1] - head[1]):
            if apple[0] < head[0]:
                if apple[1] < head[1]:
                    apples[4] = 1
                else:
                    apples[6] = 1
            else:
                if apple[1] < head[1]:
                    apples[5] = 1
                else:
                    apples[7] = 1

        # Update distances
        for i in range(len(apples)):
            if apples[i] == 1:
                walls[i] = self.dist(head, apple)


        # check if body in vision
        for i in range(len(body) - 1):
            if body[i][0] == head[0]:
                if body[i][1] < head[1]:
                    tail[2] = 1
                else:
                    tail[3] = 1
            elif body[i][1] == head[1]:
                if body[i][0] < head[0]:
                    tail[0] = 1
                else:
                    tail[1] = 1
            elif abs(body[i][0] - head[0]) == abs(body[i][1] - head[1]):
                if body[i][0] < head[0]:
                    if body[i][1] < head[1]:
                        tail[4] = 1
                    else:
                        tail[6] = 1
                else:
                    if body[i][1] < head[1]:
                        tail[5] = 1
                    else:
                        tail[7] = 1
            for j in range(len(tail)):
                if tail[j] == 1:
                    if apples[j] == 1:
                        temp = self.dist(body[i], head)
                        if walls[j] > temp:
                            walls[j] = temp
                            apples[j] = 0
                        else:
                            tail[j] = 0
                    else:
                        walls[j] = self.dist(body[i], head)

        walls.extend(apples)
        walls.extend(tail)
        #print(walls)
        input = torch.tensor(walls)



        moves = self.forward(input)
        moves = list(moves)
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

    def hypot(self, a):
        return (a ** 2 + a ** 2) ** 0.5

    def dist(self, a, b):
        xd = a[0] - b[0]
        yd = a[1] - b[1]
        return (xd**2 + yd**2)**0.5