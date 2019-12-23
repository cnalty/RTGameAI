import random
import numpy as np
import scipy.spatial as sp
import torch.nn as nn


class GeneticNet(nn.Module):
    """ This is a class that has the required class variables set up to use the genetic.py methods"""
    def __init__(self):
        self.fitness = 0
        self.diversity = 0
        self.frank
        self.drank


def set_ranks(population):
    """
    This function sets the fitness and diversity ranks for a population.

    The population given must satisfy GeneticNet class values, by either inheriting GeneticNet or having
    it's own fitness, diversity, drank and frank class variables. This method simple sets the rank, from 0 to n, where n is the size of the
    population based on each networks fitness and diversity values. Make sure these are set properly before calling this
    function.

    Parameters:
    population (list of GeneticNets): All of the GeneticNets in your population

    Returns:
    None
    """
    pop_order = sorted(population, key= lambda x: x.fitness, reverse=True)
    for i in range(len(pop_order)):
        pop_order[i].frank = i
    list.sort(pop_order, key = lambda x: x.diversity, reverse=True)
    for i in range(len(pop_order)):
        pop_order[i].drank = i


def calc_div(pop):
    """
    This function calculates the diversity of each member of the population in relation to the rest and sets their
    diversity values.

    The population given must satisfy GeneticNet class values, by either inheriting GeneticNet or having
    it's own fitness, diversity, drank and frank class variables.

    Parameters:
    pop (list of GeneticNets): All of the GeneticNets in your population

    Returns:
    None
    """
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


def get_select_distr(pop_size, pc=0.1):
    """
    This function will create a probabilistic distribution for selection.

    Parameters:
    pop_size (int): The size of your population.

    pc (float): A value between 0 and 1 for the likelyhood of your top net to be chosen, The rest of the probabilities will
                also be based on this value.

    Returns:
    unit_distr (list of float): returns a list of probabilities for each member of the population sorted from highest to
                                lowest.
    """
    distr = [pc if i == 0 else pc * (1 - pc) ** (i)
             for i in range(pop_size)]
    unit_distr = [x / sum(distr) for x in distr]
    return unit_distr



def crossover(p1, p2, network_class):
    """
    This function performs weight level crossover of two parents, returning two children. The order of parents does not
    matter.

    Parameters:
    p1 (network): First Parent
    p2 (network): Second Parent
    network_class (class): The class of the parents

    Returns:
    c1 (network): First child of parents
    c2 (network): Second child of parents
    """
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
def mutate_agents(agent_models, width, rate, bounds=(-1,1)):
    """
    Mutates each network in the population using a normal distribution of the given standard deviation and rate. The
    values of the network will be confined to the bounds given, or will be unbounded if None is given.

    Parameters:
    agent_models (network list): the population of networks to mutate
    width (float): The standard deviation for the mutation size
    rate (float): the probability of a weight being mutated
    bounds (float tuple/list or None): a pair of two numbers that will be the lower and upper bound for the networks
                                       mutations. Any mutation that changes a weight outside this range will be clipped
                                       into the range. If None is provided there will be no range bounds.

    Returns:
    None (networks are modified in place)

    """
    for agent in agent_models:
        for param in agent.parameters():
            mutate_layer(param, width, rate, bounds)


def mutate_layer(param, width, rate, bounds):
    shape = param.size()
    if len(shape) > 1:
        for sub_param in param:
            mutate_layer(sub_param, width, rate, bounds)
    else:
        for i in range(len(param)):
            if random.random() < rate:
                mutation = np.random.normal(scale=width)
                param[i] += mutation
                if bounds is not None:
                    if param[i] > bounds[1]:
                        param[i] = bounds[1]
                    elif param[i] < bounds[0]:
                        param[i] = bounds[0]


def evolve(pop, num_childs, network_class):
    """
    *Deprecated*
    performs the the entire genetic algorithm pipeline. Given a population and a number of children to produce. Uses rank
    selection.

    Parameters:
    pop (network list): all of the networks in the population
    num_childs (int): number of children to produce
    network_class (class): class of the population

    Returns:
    children (network list): the children produced

    """
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
    """
    This method performs the genetic algorithm pipeline with elitism. This uses roulette wheel selection out of the top
    num_winners members of the population and num_losers members of the population. This allows for diversity in evolution.

    Parameters:
    pop (network list): a list of all networks in the population
    num_childs (int): The number of offspring to be produced (note that this does not account for num_elite)
    num_elite (int): the number of top parents to survive to the next generation. These networks will remain completely
                     unmodified.
    num_winners (int): The number of top networks to use in roulette wheel selection
    num_losers (int): The number of worst networks to use in roulette wheel selection
    network_class (class): the class of the population

    Returns:
    kids (network list): A list of children and surviving parents

    """
    sort_pop = sorted(pop, key=lambda x: x.fitness, reverse=True)
    elite = sort_pop[:num_elite]
    parents = sort_pop[:num_winners]
    parents.extend(random.sample(sort_pop[num_winners:], num_losers))
    kids = []
    for i in range(int(num_childs / 2)):
        p1, p2 = random.sample(parents, 2)
        c1, c2 = crossover(p1, p2, network_class)
        kids.append(c1)
        if len(kids) < num_childs:
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
    pass

if __name__ == "__main__":
    main()