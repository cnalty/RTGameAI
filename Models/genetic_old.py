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
            return [torch.tensor(new)]
        elif seen < cross:
            return [p1]
        else:
            return [p2]

def crossover_2(parents, num_childs, network_class):
    children = []
    net_size = count_parameters(parents[0])
    for _ in range(num_childs):
        curr_child = network_class()
        curr_parents = random.sample(parents, 2)
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


def random_dict_crossover(parents, num_childs, network_class):
    children = []
    for _ in range(num_childs):
        curr_parents = random.sample(parents, 2)
        curr_child = network_class()
        child_state_dict = {}
        for k, v in curr_parents[0].state_dict().items():
            val1 = curr_parents[0].state_dict()[k]
            val2 = curr_parents[1].state_dict()[k]
            new_vals = random_dict_helper(val1, val2)
            #print(new_vals)
            new_torch = torch.tensor(new_vals)
            child_state_dict[k] = new_torch
        curr_child.load_state_dict(child_state_dict)
        children.append(curr_child)

    return children

def random_dict_helper(p1, p2):
    if len(p1.size()) > 1:
        size_keeper = []
        for s1, s2 in zip(p1, p2):
            size_keeper.append(random_dict_helper(s1, s2))
        return size_keeper
    else:
        vect = []
        for i in range(len(p1)):
            vect.append(random.choice([p1[i], p2[i]]))
        return vect

# crossover cut in each layer
def flat_crossover(parents, num_childs, network_class):
    children = []
    for _ in range(num_childs):
        curr_child = network_class()
        curr_parents = random.sample(parents, 2)
        for param1, param2, param3 in zip(curr_parents[0].parameters(), curr_parents[1].parameters(),
                                          curr_child.parameters()):
            layer_shape = param1.data.size()
            flat1 = param1.data.view(-1)
            flat2 = param2.data.view(-1)
            flatc = param3.data.view(-1)

            cut = random.randint(0, len(flat1))

            for j in range(len(flat1)):
                if j <= cut:
                    flatc[j] = flat1[j]
                else:
                    flatc[j] = flat2[j]

            flatc = flatc.view(layer_shape)

            param3.data = flatc
        children.append(curr_child)

    return children

''' Takes in a list of agents and a standard deviation to perform a mutation on each agent.
    Each agent receives an adjustment to all its weights given by a gaussian distribution
    with the std dev provided.'''
# Tested and working
def mutate_agents(agent_models, width, rate,bounded=True):
    for agent in agent_models:
        for param in agent.parameters():
            mutate_layer(param, width, rate, bounded)


def mutate_layer(param, width, rate, bounded):
    shape = param.size()
    if len(shape) > 1:
        for sub_param in param:
            mutate_layer(sub_param, width, rate, bounded)
    else:
        for i in range(len(param)):
            if random.random() < rate:
                mutation = np.random.normal(scale=width)
                param[i] += mutation
                if bounded:
                    if param[i] > 1:
                        param[i] = 1
                    elif param[i] < -1:
                        param[i] = -1


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def test_crossover():
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

def main():
    from Models.LookAround.look8 import LookModel8
    test_layer1 = torch.nn.Linear(4, 4)
    test_layer2 = torch.nn.Linear(4, 4)

    class test_class(torch.nn.Module):
        def __init__(self):
            super(test_class, self).__init__()
            self.fc1 = torch.nn.Linear(4, 4)


    childs = flat_crossover([test_layer1, test_layer2], 1, test_class)
    for param1, param2 in zip(test_layer1.parameters(), childs[0].parameters()):
        print(param1)
        print(param2)

if __name__ == "__main__":
    main()
