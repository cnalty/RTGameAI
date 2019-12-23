import Models.genetic as genetic
from Models.FlappyNet.flappy_net import FlappyNet
from Games.FlappyDot.world import World
import torch
import matplotlib.pyplot as plt

def main():
    flaps = [FlappyNet() for _ in range(50)]
    avg_fits = []

    for gen in range(50):
        print("-------------{}-------------".format(gen))
        for net in flaps:
            train_loop(net)

        fits = [x.fitness for x in flaps]
        print(sorted(fits, reverse=True))
        avg_fit = sum(fits) / len(fits)
        print(avg_fit)
        avg_fits.append(avg_fit)

        new_models = genetic.elite_evolve(flaps, 46, 4, 10, 1, FlappyNet)
        sorted(flaps, key=lambda x: x.fitness)
        if (gen + 1) % 10 == 0:
            torch.save(flaps[-1].state_dict(), 'flap{}.pth.tar'.format(gen))
        # print(len(new_models))
        flaps = new_models

    xs = [i for i in range(len(avg_fits))]
    plt.scatter(xs, avg_fits)


def train_loop(net):
    game = World(net.move)
    fit = game.game_loop()
    net.fitness = fit


if __name__ == "__main__":
    main()