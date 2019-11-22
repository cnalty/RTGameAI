import Models.genetic as genetic
import torch
from Models.LookAround.look import LookModel

def main():
    models = [LookModel() for _ in range(10)]
    for model in models:
        model.eval()

    for param in models[0].parameters():
        print(param.data)
        break

    genetic.mutate_agents(models, 1)
    for param in models[0].parameters():
        print(param.data)
        break


if __name__ == "__main__":
    main()
