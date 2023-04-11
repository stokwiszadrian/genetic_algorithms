import random
import networkx as nx
import pyswarms as ps
from matplotlib import pyplot as plt
from pyswarms.utils.plotters import plot_cost_history
import numpy as np
import math

# Set-up hyperparameters
options = {'c1': 0.5, 'c2': 0.2, 'w':0.8}

S = [1, 2, 3, 6, 10, 17, 25, 29, 30, 41, 51, 60, 70, 79, 80]

gene_space = [0, 1]


def generate_adjmatrix(n, prob):
    dists = [[0] * n for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if prob < random.random():
                dists[i][j] = dists[j][i] = 0
            else:
                dists[i][j] = dists[j][i] = random.randint(1, 10)

    return dists


size = 8

x_max = np.full(size, size)
x_min = np.zeros(size)
my_bounds = (x_min, x_max)
print(my_bounds)


adjmatrix = generate_adjmatrix(size, 1)
for row in adjmatrix:
    print(''.join([str(n).rjust(3, ' ') for n in row]))

# rysowanie grafu
graph = nx.Graph()
for i in range(0, size):
    for j in range(0, size):
        if adjmatrix[i][j] > 0:
            graph.add_edge(i, j, weight=adjmatrix[i][j])
pos = nx.spring_layout(graph)
nx.draw(graph, pos, with_labels=True, font_weight='bold')
labels = nx.get_edge_attributes(graph, 'weight')
nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)

# def binaryToInt(n, bits):
#     binarynum = ""
#     nums = []
#     for i in n:
#         if len(binarynum) != bits:
#             binarynum = binarynum + str(i)
#             print(binarynum)
#             if len(binarynum) == bits:
#                 num = 0
#                 for j in range(0, len(binarynum)):
#                     num += int(binarynum[j])*2**(len(binarynum)-1-j)
#                 nums.append(num)
#                 binarynum = ""
#     return nums
#
#
# print("TEST", binaryToInt([0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1], b))

v = list(map(lambda x: x - 1, [1, 2, 3, 4]))
print("TEST", v)

def f(solution_b):
    fitness = 0
    solution = np.add.reduce(solution_b, 0)
    solution = solution / np.full(size, len(solution_b))
   # print(solution)
    # for l in solution_b:
    allnodes = list(range(len(adjmatrix)))
    max_weight = 10
    solution = list(map(lambda x: math.floor(x), solution))
    #print(solution)
    # print("SOLUTION_B: ", l)
    # print("SOLUTION: ", solution)
    if solution[0] != 0:
        fitness += len(solution) * (max_weight * 10)

    for n in range(0, len(solution) - 1):

        # nodes_traveled += 1
        #
        # powtórzony node - do kosza
        # print(allnodes, solution[n], solution[n] not in allnodes)
        if solution[n] not in allnodes:
            fitness += len(solution) * (max_weight * 10)
            continue
        #
        #     # node'y nie sa polaczone - do kosza
        #     if solution[n] not in graph.adj[solution[n + 1]]:
        #         return len(solution) * (-max_weight*10)
        #
        fitness += adjmatrix[solution[n]][solution[n + 1]]
        allnodes.remove(solution[n])
        # for v in adjmatrix[n]:
        #
        #     if v == solution[n]:
        #
        #         fitness -= graph.adj[solution[n + 1]][v]['weight']
        #         allnodes.remove(solution[n])
        #         print("REMOVED", solution[n])
        #         # nagroda za powrot - im krócej, tym lepiej
        #         # if len(allnodes) == 0 and v == 0:
        #         #     return fitness + (len(solution) * max_weight - nodes_traveled * max_weight)

    if solution[-1] not in allnodes:
        fitness += len(solution) * (max_weight * 10)
    fitness += adjmatrix[solution[-1]][0]
    return fitness


optimizer = ps.single.GlobalBestPSO(n_particles=15, dimensions=size,
options=options, bounds=my_bounds)
optimizer.optimize(f, iters=10000, verbose=True)
cost_history = optimizer.cost_history
plot_cost_history(cost_history)
plt.show()

