import random
import copy
import torch
import numpy as np
import math
import scipy.spatial as sp
from scipy.stats import rv_discrete

def set_ranks(population):
    pop_order = sorted(population, key= lambda x: x.fitness, reverse=True)
    for i in range(len(pop_order)):
        pop_order[i].frank = i
    list.sort(pop_order, key = lambda x: x.diversity, reverse=True)
    for i in range(len(pop_order)):
        pop_order[i].drank = i


def calc_div(pop):
    for c1 in pop:
        for c2 in pop:
            if c1 is c2:
                continue
            else:
                for param1, param2 in zip(c1.parameters(), c2.parameters()):
                    npa1 = param1.data.numpy()
                    npa2 = param2.data.numpy()
                    npa1 = npa1.reshape(-1)
                    npa2 = npa2.reshape(-1)
                    c1.diversity += sp.distance.cosine(npa1, npa2)


def get_select_distr(pop, pc=0.1):
    '''Make sure your population list is sorted in the rank order you want before sending it to select pair'''
    distr = [pc if i == 0 else pc * (1 - pc) ** (i)
             for i in range(len(pop))]
    unit_distr = [x / sum(distr) for x in distr]
    return unit_distr



def crossover(p1, p2, network_class):
    c1 = network_class()
    c2 = network_class()
    for param1, param2, paramc1, paramc2 in zip(p1.parameters(), p2.parameters(),
                                      c1.parameters(), c2.parameters()):
        layer_shape = param1.data.size()
        flat1 = param1.data.view(-1)
        flat2 = param2.data.view(-1)
        flatc1 = paramc1.data.view(-1)
        flatc2 = paramc2.data.view(-1)

        cut = random.randint(0, len(flat1))

        for j in range(len(flat1)):
            if j <= cut:
                flatc1[j] = flat1[j]
                flatc2[j] = flat2[j]
            else:
                flatc1[j] = flat2[j]
                flatc2[j] = flat1[j]

        flatc1 = flatc1.view(layer_shape)
        flatc2 = flatc2.view(layer_shape)

        paramc1.data = flatc1
        paramc2.data = flatc2
    return c1, c2




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


def evolve(pop, num_childs, network_class):
    calc_div(pop)
    set_ranks(pop)
    sort_pop = sorted(pop, key=lambda x: x.frank + 0.5 * x.drank)
    distr = get_select_distr(sort_pop, pc=0.15)
    children = []
    for _ in range(int(num_childs / 2)):
        parents = np.random.choice(sort_pop, 2, distr)
        c1, c2 = crossover(parents[0], parents[1], network_class)
        children.append(c1)
        children.append(c2)

    mutate_agents(children, 0.5, 0.05)
    print(len(children))
    return children


def elite_evolve(pop, num_childs, num_elite, num_winners, num_losers, network_class):
    sort_pop = sorted(pop, key=lambda x: x.fitness, reverse=True)
    elite = sort_pop[:num_elite]
    parents = sort_pop[:num_winners]
    parents.extend(random.sample(sort_pop[num_winners:], num_losers))
    kids = []
    for i in range(int(num_childs / 2)):
        p1, p2 = random.sample(parents, 2)
        c1, c2 = crossover(p1, p2, network_class)
        kids.append(c1)
        kids.append(c2)
    mutate_agents(kids, 0.2, 0.02)
    kids.extend(elite)

    return kids



def test_div():
    from Models.LookAround.look8 import LookModel8
    thing1 = LookModel8()
    thing2 = LookModel8()
    things = [thing1, thing2]
    calc_div(things)
    print(thing1.diversity)

def test_distr():
    distr = get_select_distr(np.arange(20))
    print(distr)
    print(get_select_distr(np.arange(20), 0.05))
    np.random.choice(distr, 1, p=distr)

def test_crossover():
    pass


def main():
    test_distr()

if __name__ == "__main__":
    main()