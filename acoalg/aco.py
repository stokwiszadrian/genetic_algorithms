import numpy as np
from numpy import inf
import networkx as nx
import matplotlib.pyplot as plt
import random
from exact.held_karp import held_karp
from heurestic.nn import nn_tsp

# given values for the problems
def generate_adjmatrix(n, prob):
    dists = [[0] * n for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if prob < random.random():
                dists[i][j] = dists[j][i] = 0
            else:
                dists[i][j] = dists[j][i] = random.randint(1, 10)

    return dists


# size = 30
# adjmatrix = generate_adjmatrix(size, 1)
# for row in adjmatrix:
#     print(''.join([str(n).rjust(3, ' ') for n in row]))
#
# print('')
# G = nx.Graph()
# for i in range(0, size):
#     for j in range(0, size):
#         if adjmatrix[i][j] > 0:
#             G.add_edge(i, j, weight=adjmatrix[i][j])
# pos = nx.spring_layout(G)
# nx.draw(G, pos, with_labels=True, font_weight='bold')
# labels = nx.get_edge_attributes(G, 'weight')
# nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
# print("NN: ", nn_tsp(adjmatrix))
#
# nparray = np.array(adjmatrix)

def aco(d, iteration, n_ants, e, alpha, beta):

    n_cities = len(d)

    # intialization part

    m = n_ants
    n = n_cities

    # calculating the visibility of the next city visibility(i,j)=1/d(i,j)

    visibility = 1 / d
    visibility[visibility == inf] = 0

    # intializing pheromone present at the paths to the cities

    pheromone = .1 * np.ones((m, n))

    # intializing the route of the ants with size route(n_ants,n_cities+1)
    # note adding 1 because we want to come back to the source city

    route = np.ones((m, n + 1))

    for ite in range(iteration):

        route[:, 0] = 1  # initial starting and ending positon of every ants '1' i.e city '1'

        for i in range(m):

            temp_visibility = np.array(visibility)  # creating a copy of visibility

            for j in range(n - 1):
                # print(route)

                combine_feature = np.zeros(n)  # intializing combine_feature array to zero
                c_prob = np.zeros(n)  # inutializing cummulative probability array to zeros

                cur_loc = int(route[i, j] - 1)  # current city that the ant is in

                temp_visibility[:, cur_loc] = 0  # making visibility of the current city as zero

                p_feature = np.power(pheromone[cur_loc, :], alpha)  # calculating pheromone feature
                v_feature = np.power(temp_visibility[cur_loc, :], beta)  # calculating visibility feature

                p_feature = p_feature[:, np.newaxis]  # adding axis to make a size[n,1]
                v_feature = v_feature[:, np.newaxis]  # adding axis to make a size[n,1]

                combine_feature = np.multiply(p_feature, v_feature)  # calculating the combine feature

                total = np.sum(combine_feature)  # sum of all features

                probs = combine_feature / total  # finding probability of element probs(i) = combine_feature(i)/total

                c_prob = np.cumsum(probs)  # calculating cummulative sum
                # print(c_prob)
                r = np.random.random_sample()  # random float in [0,1)
                # print(r)
                city = np.nonzero(c_prob > r)[0][0] + 1  # finding the next city having probability higher then random(r)
                # print(city)

                route[i, j + 1] = city  # adding city to route

            left = list(set([i for i in range(1, n + 1)]) - set(route[i, :-2]))[0]  # finding the last untraversed city to route

            route[i, -2] = left  # adding untraversed city to route

        route_opt = np.array(route)  # intializing optimal route

        dist_cost = np.zeros((m, 1))  # intializing total_distance_of_tour with zero

        for i in range(m):

            s = 0
            for j in range(n):
                s = s + d[int(route_opt[i, j]) - 1, int(route_opt[i, j + 1]) - 1]  # calcualting total tour distance

            dist_cost[i] = s  # storing distance of tour for 'i'th ant at location 'i'

        dist_min_loc = np.argmin(dist_cost)  # finding location of minimum of dist_cost
        dist_min_cost = dist_cost[dist_min_loc]  # finding min of dist_cost

        best_route = route[dist_min_loc, :]  # initializing current traversed as best route
        pheromone = (1 - e) * pheromone  # evaporation of pheromone with (1-e)

        for i in range(m):
            for j in range(n - 1):
                dt = 1 / dist_cost[i]
                pheromone[int(route_opt[i, j]) - 1, int(route_opt[i, j + 1]) - 1] = pheromone[int(route_opt[i, j]) - 1, int(
                    route_opt[i, j + 1]) - 1] + dt
                # updating the pheromone with delta_distance

    # print('route of all the ants at the end :')
    # print(route_opt)
    print()
    def f(x):
        return x-1
    print('best path :', f(best_route))
    print('cost of the best path', int(dist_min_cost[0]) + d[int(best_route[-2]) - 1, 0])
    return int(dist_min_cost[0]) + d[int(best_route[-2]) - 1, 0]

