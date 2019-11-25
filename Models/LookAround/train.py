import Models.genetic as genetic
import torch
from Models.LookAround.look import LookModel
from Games.Snake import game
import math
import matplotlib.pyplot as plt
import threading

def main():
    gen_fitness = []
    num_threads = 10
    pop_size = 100
    curr_models = [LookModel() for _ in range(pop_size)]

    for gen in range(500):
        print("---------" + str(gen) + "---------")
        for i in range(int((len(curr_models) + 1) / num_threads)):
            threads = []
            for j in range(num_threads):
                x = threading.Thread(target=train_loop(curr_models[i*num_threads + j]))
                threads.append(x)
            for j in range(len(threads)):
                threads[j].start()
            for j in range(len(threads)):
                threads[j].join()


        fitnesses = [x.fitness for x in curr_models]
        gen_fitness.append(sum(fitnesses)/float(len(fitnesses)))
        print(fitnesses)
        winners = genetic.select_agents(fitnesses, 0.1)
        print(winners)
        win_models = [curr_models[winners[i][0]] for i in range(len(winners))]
        best = win_models[-1]
        new_models = genetic.crossover(win_models, pop_size - 1, LookModel)
        new_models.append(best)

        genetic.mutate_agents(new_models, 0.5, 0.05)
        curr_models = new_models
        xs = [i for i in range(len(gen_fitness))]
        plt.plot(xs, gen_fitness)
        plt.show()

    torch.save(curr_models[-1].state_dict(), "snake.pth.tar")

def train_loop(model):
    keys = {0: "R", 1: "L", 2: "D", 3: "U", 4: "E"}
    curr_game = game.Snake(model.choose_move, keys,
                                   "../../Games/Snake/snake.png", "../../Games/Snake/food.png",
                                    game_speed=0)
    curr_game.game_loop()



def chunk(lst, size):
    chunks = []
    for i in range(0, len(lst), size):
        chunks.append(lst[i:i + size])

    return chunks



if __name__ == "__main__":
    main()
