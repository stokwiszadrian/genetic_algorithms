def nn_tsp(G, start=None):
    cities = set(range(len(G)))
    C = start or first(cities)
    tour = [C]
    unvisited = set(cities - {C})
    while unvisited:
        C = nearest_neighbor(G, C, unvisited)
        tour.append(C)
        unvisited.remove(C)
    return tour_length(G, tour), tour


def first(collection):
    return next(iter(collection))


def nearest_neighbor(G, A, cities):
    return min(cities, key=lambda C: G[C][A])


def tour_length(G, tour):
    return sum(G[tour[i - 1]][tour[i]]
               for i in range(len(tour)))
