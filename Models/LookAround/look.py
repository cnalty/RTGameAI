import torch
import torch.nn as nn
import torch.nn.functional as F

class LookModel(nn.Module):
    def __init__(self):
        super(LookModel, self).__init__()

        # Looks in 8 directions dist to object, 2nd 8 if object is apple, 3rd 8 if object is tail
        self.fc1 = nn.Linear(8, 6)
        self.fc2 = nn.Linear(6, 6)
        #self.fc3 = nn.Linear(6, 6)
        self.fc4 = nn.Linear(6, 4) # 4 directions for output layer

        self.fitness = 0

        for param in self.parameters():
            nn.init.normal_(param.data, 0, 1)
            param.requires_grad = False

    def forward(self, x):
        out = self.fc1(x)
        out = F.leaky_relu(out)
        out = self.fc2(out)
        out = F.leaky_relu(out)
        #out = self.fc3(out)
        #out = F.leaky_relu(out)
        out = self.fc4(out)

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
        if head[0] <= 0:
            directions[1] = 0 # West Wall

            #print("wall west")
        if head[1] >= 760:
            directions[2] = 0 # South Wall

            #print("wall south")
        if head[1] <= 0:
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


        for i in range(len(moves)):
            if moves[i] and directions[i] == 0:
                self.fitness -= 100
            elif not moves[i] and directions[i] == 0:
                self.fitness += 100

            if not moves[i] and directions[i + 4] == 1:
                self.fitness -= 1000

            elif directions[i + 4] == 1:
                self.fitness += 2000


        moves.append(False)
        return moves

    def hypot(self, a):
        return (a ** 2 + a ** 2) ** 0.5

    def dist(self, a, b):
        xd = a[0] - b[0]
        yd = a[1] - b[1]
        return (xd**2 + yd**2)**0.5

    def fitness_model(self, fitness_params):
        score = fitness_params[0]
        closeness = fitness_params[1]
        turns = fitness_params[2]
        away = fitness_params[3]

        fitness = score ** 4

        fitness += turns

        fitness -= away * 1.5

        return fitness

    def test_move(self, image, body ,apple):
        choices = [True for i in range(5)]
        head = body[-1]
        directions = [1 if i < 4 else 0 for i in range(8)]
        for i in range(len(body) - 1):
            if body[i][0] == head[0] + 40:
                if body[i][1] == head[1]:
                    choices[1] = False
                    # print("body east")
                    directions[0] = 0  # East
            elif body[i][0] == head[0] - 40:
                if body[i][1] == head[1]:
                    choices[0] = False
                    directions[1] = 0  # West
                    # print("body west")
            elif body[i][1] == head[1] + 40:
                if body[i][0] == head[0]:
                    choices[3] = False
                    directions[2] = 0  # South
                    # print("body south")
            elif body[i][1] == head[1] - 40:
                if body[i][0] == head[0]:
                    choices[2] = False
                    directions[3] = 0  # North
                    # print("body north")

        if head[0] >= 760:
            directions[0] = 0  # East Wall
            choices[1] = False
            # print("wall east")
        if head[0] <= 0:
            directions[1] = 0  # West Wall
            choices[0] = False
            # print("wall west")
        if head[1] >= 760:
            directions[2] = 0  # South Wall
            choices[3] = False
            # print("wall south")
        if head[1] <= 0:
            directions[3] = 0  # North Wall
            choices[2] = False
            # print("wall north")

        if apple[1] == head[1]:
            if head[0] < apple[0]:
                directions[4] = 1
                # print("apple east")
            else:
                directions[5] = 1
                # print("apple west")
        elif apple[0] == head[0]:
            if head[1] < apple[1]:
                directions[6] = 1
                # print("apple south")
            else:
                directions[7] = 1
                # print("apple north")

        return choices