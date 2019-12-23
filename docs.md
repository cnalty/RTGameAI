Module Models.genetic
=====================

Functions
---------


`calc_div(pop)`
:   This function calculates the diversity of each member of the population in relation to the rest and sets their
    diversity values.

    The population given must satisfy GeneticNet class values, by either inheriting GeneticNet or having
    it's own fitness, diversity, drank and frank class variables.

    Parameters:
    pop (list of GeneticNets): All of the GeneticNets in your population

    Returns:
    None


`crossover(p1, p2, network_class)`
:   This function performs weight level crossover of two parents, returning two children. The order of parents does not
    matter.

    Parameters:
    p1 (network): First Parent
    p2 (network): Second Parent
    network_class (class): The class of the parents

    Returns:
    c1 (network): First child of parents
    c2 (network): Second child of parents


`elite_evolve(pop, num_childs, num_elite, num_winners, num_losers, network_class)`
:   This method performs the genetic algorithm pipeline with elitism. This uses roulette wheel selection out of the top
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


`evolve(pop, num_childs, network_class)`
:   *Deprecated*
    performs the the entire genetic algorithm pipeline. Given a population and a number of children to produce. Uses rank
    selection.

    Parameters:
    pop (network list): all of the networks in the population
    num_childs (int): number of children to produce
    network_class (class): class of the population

    Returns:
    children (network list): the children produced


`get_select_distr(pop_size, pc=0.1)`
:   This function will create a probabilistic distribution for selection.

    Parameters:
    pop_size (int): The size of your population.

    pc (float): A value between 0 and 1 for the likelyhood of your top net to be chosen, The rest of the probabilities will
                also be based on this value.

    Returns:
    unit_distr (list of float): returns a list of probabilities for each member of the population sorted from highest to
                                lowest.


`mutate_agents(agent_models, width, rate, bounds=(-1, 1))`
:   Mutates each network in the population using a normal distribution of the given standard deviation and rate. The
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



`set_ranks(population)`
:   This function sets the fitness and diversity ranks for a population.

    The population given must satisfy GeneticNet class values, by either inheriting GeneticNet or having
    it's own fitness, diversity, drank and frank class variables. This method simple sets the rank, from 0 to n, where n is the size of the
    population based on each networks fitness and diversity values. Make sure these are set properly before calling this
    function.

    Parameters:
    population (list of GeneticNets): All of the GeneticNets in your population

    Returns:
    None


Classes
-------

`GeneticNet()`
:   This is a class that has the required class variables set up to use the genetic.py methods

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module
