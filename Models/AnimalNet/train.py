import Models.genetic2 as genetic
from Models.AnimalNet.animal_net import AnimalNet
from Games.HungrySquares.world import World
import matplotlib.pyplot as plt

def main():
    animals = [AnimalNet() for _ in range(50)]


    for gen in range(50):
        print("-------------{}-------------".format(gen))
        fits = train_loop(animals)
        print(sorted(fits, reverse=True))

        new_models = genetic.elite_evolve(animals, 46, 4, 10, 1, AnimalNet)
        #print(len(new_models))
        animals = new_models


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