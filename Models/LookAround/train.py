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
    curr_models = [LookModel() for _ in range(50)]

    for gen in range(20):
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
        actual_fit = [fitness_model(x) for x in fitnesses]
        gen_fitness.append(sum(actual_fit)/float(len(actual_fit)))
        print(fitnesses)
        winners = genetic.select_agents(fitnesses, fitness_model, 0.2)
        print(winners)
        win_models = [curr_models[winners[i][0]] for i in range(len(winners))]
        new_models = genetic.crossover2(win_models, 5)

        genetic.mutate_agents(new_models, 5, 0.5)
        curr_models = new_models

    xs = [i for i in range(len(gen_fitness))]
    plt.plot(xs, gen_fitness)
    plt.show()

def train_loop(model):
    keys = {0: "L", 1: "R", 2: "U", 3: "D", 4: "E"}
    curr_game = game.Snake(model.choose_move, keys,
                                   "../../Games/Snake/snake.png", "../../Games/Snake/food.png",
                                    game_speed=0)
    model.fitness = curr_game.game_loop()



def chunk(lst, size):
    chunks = []
    for i in range(0, len(lst), size):
        chunks.append(lst[i:i + size])

    return chunks

def fitness_model(fitness_params):
    score = fitness_params[0]
    closeness = fitness_params[1]
    turns = fitness_params[2]
    away = fitness_params[3]

    fitness = score ** 5

    fitness += turns

    fitness -= away

    return fitness






if __name__ == "__main__":
    main()
