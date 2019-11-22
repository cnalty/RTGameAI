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
            curr_score = fitness_model(agent_results)
            if curr_score > best_agents[0][1]:
                list.remove(0)
                list.append((i, curr_score))

    return best_agents

def crossover_agents(agent_models, times_pair):
    new_models = []
    for i in range(times_pair):
        curr_models = copy.deepcopy(agent_models)
        random.shuffle(curr_models)

        for j in range(1, len(curr_models), 2):
            n1, p1 = curr_models[j].named_parameters()
            n2, p2 = curr_models[j - 1].named_parameters()

            for k in range(1, len(p1), 2):
                temp = p1[k]
                p1[k] = p2[k]
                p2[k] = temp

            new_models.append(copy.deepcopy(curr_models[j]))
            new_models.append(copy.deepcopy(curr_models[j - 1]))


''' Takes in a list of agents and a standard deviation to perform a mutation on each agent.
    Each agent receives an adjustment to all its weights given by a gaussian distribution
    with the std dev provided.'''
# Tested and working
def mutate_agents(agent_models, width):
    for agent in agent_models:
        curr_params = agent.parameters()
        for param in agent.parameters():
            shape = param.size()

            if len(shape) > 1:
                for sub_param in param:
                    size = len(sub_param)
                    mutation = np.random.normal(scale=width, size=size)
                    param += torch.tensor(mutation)
            else:
                size = len(param)
                mutation = np.random.normal(scale=width, size=size)
                param += torch.tensor(mutation)

