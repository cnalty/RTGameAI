import numpy as np
import networkx as nx
from numpy.random import uniform

class NEATwork():
    INNOVATION = 0
    def __init__(self, inputs, outputs):
        self.genes = []
        self.layers = []
        self.network = nx.Graph()
        self.in_nodes = set()
        self.out_nodes = set()
        for i in range(inputs):
            self.in_nodes.add(i)
            for j in range(inputs, outputs + inputs):
                self.out_nodes.add(j)
                self.network.add_edge(i, j, weight=uniform(-1, 1))

    ''' Going to start from the output nodes and go backwards'''
    def forward(self, x):
        pass

    def similarity(self, net2):
        c1 = 1
        c2 = 1
        c3 = 1
        N = max(len(self), len(net2))
        E = len(self.get_excess())
        D = len(self.get_disjoint())
        W = 0
        i = 0
        j = 0
        while i < len(self) and j < len(net2):
            if self.genes[i].innov_num == self.genes[i].innov_num:
                W += abs(self.nodes[i].weight - net2.nodes[j].weight)
            elif self.genes[i].innov_num < self.genes[i].innov_num:
                i += 1
            else:
                j += 1
        W /= N

        delta = c1 * E / N + c2 * D / N + c3 * W

        return delta

    def get_excess(self, net2):
        pass

    def get_disjoint(self, net2):
        pass

    def __len__(self):
        return len(self.nodes)

class neat_gene():
    ''' When adding a new node is made by splitting a connection the weight into the node becomes 1 and the output
        weight is the old weight'''
    def __init__(self, in_node, out_node, innov_num):
        self.innov_num = innov_num
        self.in_node = in_node
        self.out_node = out_node
        self.weight = 1