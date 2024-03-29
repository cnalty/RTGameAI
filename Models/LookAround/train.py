import Models.genetic as genetic
import torch
from Models.LookAround.look import LookModel
from Models.LookAround.look8 import LookModel8
from Games.Snake import game
import math
import matplotlib.pyplot as plt
import threading

def genetic_train():
    gen_fitness = []
    num_threads = 10
    pop_size = 100
    curr_models = [LookModel8() for _ in range(pop_size)]

    for gen in range(100):
        print("---------" + str(gen) + "---------")
        for i in range(len(curr_models)):
           train_loop(curr_models[i])

        fitnesses = [x.fitness for x in curr_models]
        gen_fitness.append(sum(fitnesses) / len(fitnesses))
        print(sum(fitnesses)/len(fitnesses))

        new_models = genetic.elite_evolve(curr_models, 95, 5, 20, 10, LookModel8)
        curr_models = new_models

    xs = [i for i in range(len(gen_fitness))]
    torch.save(curr_models[-1].state_dict(), "snake.pth.tar")
    plt.plot(xs, gen_fitness)
    plt.show()


def train_loop(model):
    keys = {0: "R", 1: "L", 2: "D", 3: "U", 4: "E"}
    curr_game = game.Snake(model.choose_move, keys,
                                   "../../Games/Snake/snake.png", "../../Games/Snake/food.png",
                                    game_speed=0)
    model.fitness = model.fitness_model(curr_game.game_loop())



def chunk(lst, size):
    chunks = []
    for i in range(0, len(lst), size):
        chunks.append(lst[i:i + size])

    return chunks



if __name__ == "__main__":
    genetic_train()
