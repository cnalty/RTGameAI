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
                    c1.diversity += sp.distance.cdist(npa1, npa2, 'cosine')


def get_select_distr(pop, pc=0.05):
    '''Make sure your population list is sorted in the rank order you want before sending it to select pair'''
    distr = [pc if i == 1 else (1 - pc) ** (len(pop) - 1) if i == len(pop) - 1 else pc * (1 - pc) ** (i - 1)
             for i in range(len(pop))]
    return distr

def select_pair(distr):
    pass