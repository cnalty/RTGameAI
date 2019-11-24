import torch
import torch.nn as nn
import torch.nn.functional as F

class LookModel(nn.Module):
    def __init__(self):
        super(LookModel, self).__init__()

        # Looks in 8 directions dist to object, 2nd 8 if object is apple, 3rd 8 if object is tail
        self.fc1 = nn.Linear(8, 6)
        nn.init.uniform_(self.fc1.weight.data, -100, 100)
        self.fc2 = nn.Linear(6, 6)
        nn.init.uniform_(self.fc2.weight.data, -100, 100)
        self.fc3 = nn.Linear(6, 4) # 4 directions for output layer
        nn.init.uniform_(self.fc3.weight.data, -100, 100)
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
        directions = [1 if i < 4 else 0 for i in range(8)]
        for i in range(len(body) - 1):
            if body[i][0] == head[0] + 40:
               if body[i][1] == head[1]:
                   #print("body east")
                   directions[0] = 0 # East
            elif body[i][0] == head[0] - 40:
                if body[i][1] == head[1]:
                    directions[1] = 0 # West
                    #print("body west")
            elif body[i][1] == head[1] + 40:
               if body[i][0] == head[0]:
                   directions[2] = 0 # South
                   #print("body south")
            elif body[i][1] == head[1] - 40:
                if body[i][0] == head[0]:
                    directions[3] = 0 # North
                    #print("body north")

        if head[0] >= 760:
            directions[0] = 0 # East Wall
            #print("wall east")
        elif head[0] <= 0:
            directions[1] = 0 # West Wall
            #print("wall west")
        elif head[1] >= 760:
            directions[2] = 0 # South Wall
            #print("wall south")
        elif head[1] <= 0:
            directions[3] = 0 # North Wall
            #print("wall north")

        if apple[1] == head[1]:
            if head[0] < apple[0]:
                directions[4] = 1
                #print("apple east")
            else:
                directions[5] = 1
                #print("apple west")
        elif apple[0] == head[0]:
            if head[1] < apple[1]:
                directions[6] = 1
                #print("apple south")
            else:
                directions[7] = 1
                #print("apple north")
        #print("--------------")

        #print(directions)
        input = torch.tensor(directions).float()
        #print(input)

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