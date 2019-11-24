import random
import copy
import torch
import numpy as np

def select_agents(agent_results, fitness_model, percent):
    num_agents = int(len(agent_results) * percent)
    best_agents = []
    for i in range(len(agent_results)):
        if len(best_agents) < num_agents:
            best_agents.append((i, fitness_model(agent_results[i])))
        else:
            list.sort(best_agents, key=lambda x: x[1])
            curr_score = fitness_model(agent_results[i])
            if curr_score > best_agents[0][1]:
                best_agents.pop(0)
                best_agents.append((i, curr_score))

    return best_agents

def crossover_agents(agent_models, times_pair=2):
    new_models = []
    for i in range(times_pair):
        curr_models = copy.deepcopy(agent_models)
        random.shuffle(curr_models)

        for j in range(1, len(curr_models), 2):
            i = 0
            for param1, param2 in zip(curr_models[j].parameters(), curr_models[j - 1].parameters()):
                if i % 2 == 0:
                    #print("swap")
                    temp = param1.data
                    param1.data = param2.data
                    param2.data = temp
                i += 1
            new_models.append(curr_models[j - 1])
            new_models.append(curr_models[j])



    return new_models

def crossover2(agent_models, times_pair=2):
    new_models = []
    for i in range(times_pair):
        curr_models = copy.deepcopy(agent_models)
        random.shuffle(curr_models)

        for j in range(1, len(curr_models), 2):
            i = 0
            for param1, param2 in zip(curr_models[j].parameters(), curr_models[j - 1].parameters()):
                swap_point = random.randint(0, len(param1.data) - 1)
                for i in range(swap_point):
                    param1.data[i] = param2.data[i]
                for i in range(swap_point, len(param1.data)):
                    param2.data[i] = param1.data[i]
            new_models.append(curr_models[j - 1])
            new_models.append(curr_models[j])

    return new_models



''' Takes in a list of agents and a standard deviation to perform a mutation on each agent.
    Each agent receives an adjustment to all its weights given by a gaussian distribution
    with the std dev provided.'''
# Tested and working
def mutate_agents(agent_models, width, rate):
    for agent in agent_models:
        curr_params = agent.parameters()
        for param in agent.parameters():
            if random.random() > rate:
                continue
            shape = param.size()

            if len(shape) > 1:
                for sub_param in param:
                    size = len(sub_param)
                    mutation = np.random.normal(scale=width, size=size)
                    param += torch.tensor(mutation).float()
            else:
                size = len(param)
                mutation = np.random.normal(scale=width, size=size)
                param += torch.tensor(mutation).float()

