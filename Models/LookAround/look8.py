import torch
import torch.nn as nn
import torch.nn.functional as F


class LookModel8(nn.Module):
    def __init__(self):
        super(LookModel8, self).__init__()

        # Looks in 8 directions dist to object, 2nd 8 if object is apple, 3rd 8 if object is tail
        self.fc1 = nn.Linear(24, 18)
        self.fc2 = nn.Linear(18, 18)
        self.fc4 = nn.Linear(18, 4) # 4 directions for output layer

        self.fitness = 0

        for param in self.parameters():
            nn.init.normal_(param.data, 0, 1)
            param.requires_grad = False

    def forward(self, x):
        out = self.fc1(x)
        out = F.leaky_relu(out)
        out = self.fc2(out)
        out = F.leaky_relu(out)
        out = self.fc4(out)

        return out

    def choose_move(self, image, body, apple):
        directions = []
        move_size = 40
        move_dirs = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (-1, -1), (1, 1)]
        for dir in move_dirs:
            saw = self.look(dir, move_size, body, apple)
            directions.extend(saw)

        #print(directions)
        input = torch.tensor(directions)

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

    def look(self, dir, move_size, body, apple):

        pos = list(body[-1])
        dist = 1
        max = 20
        while(pos[0] > 0 and pos[0] <= 800 and pos[1] > 0 and pos[1] <= 800):
            pos[0] += dir[0] * move_size
            pos[1] += dir[1] * move_size
            for i in range(len(body) - 1):
                if body[i][0] == pos[0] and body[i][1] == pos[1]:
                    return [ dist / max, 1, 0]
            if apple[0] == pos[0] and apple[1] == pos[1]:
                return [dist / max, 0, 1]
            dist += 1
        return [dist / max, 0, 0]

    def fitness_model(self, fitness_params):
        score = fitness_params[0]
        closeness = fitness_params[1]
        turns = fitness_params[2]
        away = fitness_params[3]

        fitness = 20 * (score - 3)

        fitness += turns * 4

        #fitness -= away * away

        return fitness

