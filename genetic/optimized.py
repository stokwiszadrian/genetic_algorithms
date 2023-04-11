import pygad
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import time
from exact.held_karp import held_karp
from heurestic.nn import nn_tsp
from acoalg.aco import aco

if __name__ == "__main__":


    # graph = nx.fast_gnp_random_graph(10, 1, 666)
    # for (u, v, w) in graph.edges(data=True):
    #     w['weight'] = random.randint(1, 100)
    # pos = nx.spring_layout(graph)
    # nx.draw(graph, pos, with_labels=True, font_weight='bold')
    # labels = nx.get_edge_attributes(graph, 'weight')
    # nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
    # print(10 in graph.adj[0].keys())
    # print(graph.nodes)
    # print(graph.adj[0])
    # for i in graph.adj[0]:
    #     print(i, graph.adj[0][i]['weight'])

    # nx.draw_shell(graph, nlist=[range(5, 10), range(5)], with_labels=True, font_weight='bold')
    # test = nx.adjacency_matrix(graph, weight="weight")
    # print(nx.to_dict_of_dicts(test))

    #print("HELD-KARP")
    #print(held_karp(adjmatrix))

    # 0 MUSI być wezlem poczatkowym
    def fitness_func1(solution, solution_idx):
        fitness = 0
        allnodes = list(range(len(adjmatrix)))
        max_weight = 10

        # sprawdzenie, czy węzeł początkowy jest 0em
        if solution[0] != 0:
            return len(solution) * (-max_weight * 10)

        for n in range(0, len(solution) - 1):

            # powtórzony node - konczy działanie funkcji
            # zwraca dotychczasowy fitness i karę na podstawie pozostałych wezlow

            if solution[n] not in allnodes:
                return fitness - len(allnodes)*(max_weight * 10)

            fitness -= adjmatrix[solution[n]][solution[n + 1]]
            allnodes.remove(solution[n])

        if solution[-1] not in allnodes:
            return fitness - len(allnodes)*(max_weight * 10)

        return fitness - adjmatrix[solution[-1]][0]

    # bez wskazanego 1szego wezla
    def fitness_func2(solution, solution_idx):
        fitness = 0
        allnodes = list(range(len(adjmatrix)))
        max_weight = 10

        for n in range(0, len(solution) - 1):

            # powtórzony node - konczy działanie funkcji
            # zwraca dotychczasowy fitness i karę na podstawie pozostałych wezlow

            if solution[n] not in allnodes:
                return fitness - len(allnodes)*(max_weight * 10)

            fitness -= adjmatrix[solution[n]][solution[n + 1]]
            allnodes.remove(solution[n])

        if solution[-1] not in allnodes:
            return fitness - len(allnodes)*(max_weight * 10)

        return fitness - adjmatrix[solution[-1]][solution[0]]


    size = 30

    def generate_adjmatrix(n, prob):
        m = [[0] * n for i in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if prob < random.random():
                    m[i][j] = m[j][i] = 0
                else:
                    m[i][j] = m[j][i] = random.randint(1, 10)
        return m

    testmatrix = [
        [0, 7, 2, 10, 1, 2, 6, 4, 7, 6, 2, 7, 8, 9, 9, 5, 4, 10, 9, 1],
        [7, 0, 8, 7, 9, 4, 7, 4, 8, 8, 4, 7, 10, 7, 5, 9, 4, 8, 9, 3],
        [2, 8, 0, 4, 1, 7, 4, 6, 4, 4, 6, 1, 2, 1, 7, 4, 3, 1, 10, 1],
        [10, 7, 4, 0, 7, 7, 6, 9, 4, 3, 9, 6, 8, 8, 4, 7, 7, 7, 9, 9],
        [1, 9, 1, 7, 0, 7, 9, 10, 7, 1, 8, 8, 2, 1, 10, 4, 5, 9, 2, 6],
        [2, 4, 7, 7, 7, 0, 4, 5, 5, 3, 2, 3, 2, 3, 9, 6, 10, 10, 1, 10],
        [6, 7, 4, 6, 9, 4, 0, 6, 2, 3, 6, 2, 5, 10, 5, 8, 1, 6, 9, 2],
        [4, 4, 6, 9, 10, 5, 6, 0, 8, 3, 5, 6, 9, 5, 4, 7, 7, 2, 7, 7],
        [7, 8, 4, 4, 7, 5, 2, 8, 0, 6, 2, 10, 5, 6, 3, 7, 1, 8, 7, 2],
        [6, 8, 4, 3, 1, 3, 3, 3, 6, 0, 9, 2, 7, 9, 10, 3, 5, 6, 10, 8],
        [2, 4, 6, 9, 8, 2, 6, 5, 2, 9, 0, 10, 2, 5, 9, 5, 7, 2, 1, 1],
        [7, 7, 1, 6, 8, 3, 2, 6, 10, 2, 10, 0, 5, 2, 10, 8, 1, 1, 4, 6],
        [8, 10, 2, 8, 2, 2, 5, 9, 5, 7, 2, 5, 0, 5, 9, 7, 6, 9, 5, 6],
        [9, 7, 1, 8, 1, 3, 10, 5, 6, 9, 5, 2, 5, 0, 9, 9, 6, 9, 6, 5],
        [9, 5, 7, 4, 10, 9, 5, 4, 3, 10, 9, 10, 9, 9, 0, 6, 1, 3, 5, 4],
        [5, 9, 4, 7, 4, 6, 8, 7, 7, 3, 5, 8, 7, 9, 6, 0, 6, 4, 2, 3],
        [4, 4, 3, 7, 5, 10, 1, 7, 1, 5, 7, 1, 6, 6, 1, 6, 0, 8, 8, 7],
        [10, 8, 1, 7, 9, 10, 6, 2, 8, 6, 2, 1, 9, 9, 3, 4, 8, 0, 10, 1],
        [9, 9, 10, 9, 2, 1, 9, 7, 7, 10, 1, 4, 5, 6, 5, 2, 8, 10, 0, 1],
        [1, 3, 1, 9, 6, 10, 2, 7, 2, 8, 1, 6, 6, 5, 4, 3, 7, 1, 1, 0]]

    adjmatrix = generate_adjmatrix(size, 1)
    print(adjmatrix)
    # for i in adjmatrix:
    #     print(i)
    # for row in adjmatrix:
    #     print(''.join([str(n).rjust(3, ' ') for n in row]))
    # start = time.time()
    # print("HELD-KARP", held_karp(adjmatrix))
    # stop= time.time()

    # rysowanie grafu
    graph = nx.Graph()
    for i in range(0, size):
        for j in range(0, size):
            if adjmatrix[i][j] > 0:
                graph.add_edge(i, j, weight=adjmatrix[i][j])
    # pos = nx.spring_layout(graph)
    # nx.draw(graph, pos, with_labels=True, font_weight='bold')
    # labels = nx.get_edge_attributes(graph, 'weight')
    # nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)

    gene_space = list(graph.nodes)
    sol_per_pop = 50
    num_genes = len(adjmatrix)

    num_parents_mating = 25
    num_generations = 3500
    keep_parents = 4

    parent_selection_type = "sss"

    crossover_type = "single_point"

    mutation_type = "random"
    mutation_percent_genes = int(120 / num_genes)

    #func1
    avgfunc2 = 0
    avgaco = 0
    avgtime2 = 0
    avgtimeaco = 0

    iters = 5

    print("NN")
    tour_length, tour = nn_tsp(adjmatrix)
    print(tour_length, tour)

    # start = time.time()
    # print("HELD-KARP:")
    # print(held_karp(adjmatrix))
    # stop = time.time()
    # print("KARP TIME: ", stop - start)


    print("FUNC2")
    for i in range(0, iters):

        # inicjacja algorytmu z powyzszymi parametrami wpisanymi w atrybuty
        ga_instance = pygad.GA(gene_space=gene_space,
                               gene_type=int,
                               num_generations=num_generations,
                               num_parents_mating=num_parents_mating,
                               fitness_func=fitness_func2,
                               sol_per_pop=sol_per_pop,
                               num_genes=num_genes,
                               parent_selection_type=parent_selection_type,
                               keep_parents=keep_parents,
                               crossover_type=crossover_type,
                               mutation_type=mutation_type,
                               mutation_percent_genes=mutation_percent_genes)

        start = time.time()
        # uruchomienie algorytmu
        ga_instance.run()
        stop = time.time()
        elapsed = stop - start
        # podsumowanie: najlepsze znalezione rozwiazanie (chromosom+ocena)
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        print("Parameters of the best solution : {solution}".format(solution=solution))
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
        check = tour
        for i in range(size):
            if i not in solution:
                print("INCORRECT SOLUTION:", i, "MISSING")
        print("TIME ELAPSED: ", elapsed)
        avgfunc2 += solution_fitness
        avgtime2 += elapsed

        start = time.time()
        acoscore = aco(np.array(adjmatrix), 300, 50, 0.5, 1, 2)
        stop = time.time()
        elapsed = stop - start
        avgtimeaco += elapsed
        avgaco += acoscore


    print(avgfunc2/iters, avgaco/iters)
    print(avgtime2 / iters, avgtimeaco / iters)
    plt.show()
