import Models.genetic2 as genetic
from Models.AnimalNet.animal_net import AnimalNet
from Games.HungrySquares.world import World
import torch
import matplotlib.pyplot as plt

def main():
    animals = [AnimalNet() for _ in range(50)]
    avg_fits = []

    for gen in range(20):
        print("-------------{}-------------".format(gen))
        fits = train_loop(animals)
        print(sorted(fits, reverse=True))
        avg_fit = sum(fits) / len(fits)
        print(avg_fit)
        avg_fits.append(avg_fit)


        new_models = genetic.elite_evolve(animals, 46, 4, 10, 1, AnimalNet)
        sorted(animals, key=lambda x: x.fitness)
        if (gen + 1) % 10 == 0:
            torch.save(animals[-1].state_dict(), 'animal{}.pth.tar'.format(gen))
        #print(len(new_models))
        animals = new_models

    xs = [i for i in range(len(avg_fits))]
    plt.scatter(xs, avg_fits)

def train_loop(nets):
    movers = [net.steer for net in nets]
    game1 = World(movers[:25])
    fits = game1.game_loop()
    game2 = World(movers[25:])
    fits.extend(game2.game_loop())
    for i in range(len(fits)):
        nets[i].fitness = fits[i]

    return fits


if __name__ == "__main__":
    main()