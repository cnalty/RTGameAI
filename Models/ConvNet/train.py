import Models.genetic as genetic
import torch
from Models.ConvNet.convnet import ConvNet
from Games.Snake import game
import math
import matplotlib.pyplot as plt
import threading

def main():
    gen_fitness = []
    num_threads = 10
    pop_size = 50
    curr_models = [ConvNet() for _ in range(pop_size)]

    for gen in range(200):
        print("---------" + str(gen) + "---------")
        for i in range(int((len(curr_models) + 1) / num_threads)):
            threads = []
            for j in range(num_threads):
                x = threading.Thread(target=train_loop(curr_models[i * num_threads + j]))
                threads.append(x)
            for j in range(len(threads)):
                threads[j].start()
            for j in range(len(threads)):
                threads[j].join()

        fitnesses = [x.fitness for x in curr_models]
        gen_fitness.append(sum(fitnesses) / float(len(fitnesses)))
        print(fitnesses)
        winners = genetic.select_agents(fitnesses, 0.2)
        print(winners)
        win_models = [curr_models[winners[i][0]] for i in range(len(winners))]
        new_models = genetic.crossover(win_models, pop_size, ConvNet)

        genetic.mutate_agents(new_models, .5, 0.5)
        curr_models = new_models

    xs = [i for i in range(len(gen_fitness))]
    plt.plot(xs, gen_fitness)
    plt.show()
    torch.save(curr_models[-1].state_dict(), "snake.pth.tar")


def train_loop(model):
    keys = {0: "L", 1: "R", 2: "U", 3: "D", 4: "E"}
    curr_game = game.Snake(model.choose_move, keys,
                                   "../../Games/Snake/snake.png", "../../Games/Snake/food.png",
                                    game_speed=0)
    model.fitness = model.fitness_model(curr_game.game_loop())

if __name__ == "__main__":
    main()