import Models.genetic as genetic
import torch
from Models.LookAround.look import LookModel

def main():
    models = [LookModel() for _ in range(10)]
    i = 1
    for param in models[0].parameters():
        print(param)
        if i % 2 == 0:
            break
        i += 1
    i += 1
    new_models = genetic.crossover_agents(models, 1)
    for param in new_models[0].parameters():
        print(param)
        if i % 2 == 0:
            break
        i += 1




if __name__ == "__main__":
    main()
