import random
import copy
import torch
import numpy as np
import math

def select_agents(fits, percent):
    num_agents = math.ceil(len(fits) * percent)
    best_agents = []
    for i in range(len(fits)):
        if len(best_agents) < num_agents:
            best_agents.append((i, fits[i]))
        else:
            list.sort(best_agents, key=lambda x: x[1])
            if fits[i] > best_agents[0][1]:
                best_agents.pop(0)
                best_agents.append((i, fits[i]))

    list.sort(best_agents, key=lambda  x: x[1])
    return best_agents

def agent_distribution(fits):
    total_fit = 0
    for x in fits:
        total_fit += x ** 2
    distr = [(x ** 2) / total_fit for x in fits]
    #print(sum(distr))

    return distr

def crossover(parents, num_childs, network_class, distr=None):
    children = []
    net_size = count_parameters(parents[0])
    for _ in range(num_childs):
        curr_child = network_class()
        if distr is None:
            curr_parents = random.sample(parents, 2)

        else:
            curr_parents = np.random.choice(parents, 2, replace=False, p=distr)
            #print(curr_parents)

        cross_point = random.randint(1, net_size - 1)
        start_n = 0
        for param1, param2, param3 in zip(curr_parents[0].parameters(), curr_parents[1].parameters(),
                                          curr_child.parameters()):

            datas = crossover_weight(param1, param2, param3, start_n, cross_point)
            if len(datas) > 1:
                param3.data = torch.stack(datas)
            else:
                param3.data = datas[0]

            curr_dims = param1.size()
            curr_size = 1
            for x in curr_dims:
                curr_size *= x
            start_n += curr_size
        children.append(curr_child)

    return children

def crossover_weight(p1, p2, c, seen, cross):
    shape = p1.size()
    if len(shape) > 1:
        datas = []
        for s1, s2, sc in zip(p1, p2, c):
            datas.extend(crossover_weight(s1, s2, sc, seen, cross))
            seen += len(sc)
        return datas
    else:
        if seen < cross and seen + len(c) > cross:
            #print("crossed")
            cross_p = cross - seen
            new = []
            for i in range(len(p1)):
                if i < cross_p:
                    new.append(p1[i])
                else:
                    new.append(p2[i])
            return [torch.tensor(new).cuda()]
        elif seen < cross:
            return [p1]
        else:
            return [p2]

def crossover_2(parents, num_childs, network_class):
    children = []
    net_size = count_parameters(parents[0])
    for _ in range(num_childs):
        curr_child = network_class()
        curr_parents = np.random.choice(parents, 2)
        for param1, param2, param3 in zip(curr_parents[0].parameters(), curr_parents[1].parameters(),
                                          curr_child.parameters()):
            datas = crossover_layer(param1, param2, param3)
            if len(datas) > 1:
                param3.data = torch.stack(datas)
            else:
                param3.data = datas[0]
        children.append(curr_child)

    return children

def crossover_layer(p1, p2, c):
    shape = p1.size()
    if len(shape) > 1:
        datas = []
        for s1, s2, sc in zip(p1, p2, c):
            datas.extend(crossover_layer(s1, s2, sc))
        return datas
    else:
        c = random.choice([p1.data, p2.data])
        #print(c)
        return [c]

''' Takes in a list of agents and a standard deviation to perform a mutation on each agent.
    Each agent receives an adjustment to all its weights given by a gaussian distribution
    with the std dev provided.'''
# Tested and working
def mutate_agents(agent_models, width, rate):
    for agent in agent_models:
        for param in agent.parameters():
            mutate_layer(param, width, rate)


def mutate_layer(param, width, rate):
    shape = param.size()
    if len(shape) > 1:
        for sub_param in param:
            mutate_layer(sub_param, width, rate)
    else:
        for i in range(len(param)):
            if random.random() < rate:
                mutation = np.random.normal(scale=width)
                param[i] += mutation


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def main():
    test_layer1 = torch.nn.Linear(10, 2)
    test_layer2 = torch.nn.Linear(10, 2)
    test_child = torch.nn.Linear(10, 2)

    for param in test_layer1.parameters():
        print(param.data)
        print("---------")

    # for param in test_layer2.parameters():
    #     print(param.data)
    #     print("---------")

    for param1, param2, param3 in zip(test_layer1.parameters(), test_layer2.parameters(),
                                      test_child.parameters()):
        datas = crossover_weight(param1, param2, param3, 0, 5)

        if len(datas) > 1:
            param3.data = torch.stack(datas)
        else:
            param3.data = datas[0]


    for param in test_child.parameters():
        print(param.data)


if __name__ == "__main__":
    main()
