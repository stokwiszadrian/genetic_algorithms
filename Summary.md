# Zadanie projektowe nr 1

# Zadanie projektowe nr 1
### Algorytmy genetyczne i roje cząstek

## 1. Temat projektu

Wybranym przeze mnie problemem do rozwiązania jest **Problem komiwojażera (ang. travelling salesman problem, TSP)**. W tym problemie staramy się odpowiedzieć na następujące pytanie: *Mając daną miast i odległości między każdą parą miast, jaka jest najkrótsza możliwa trasa, która odwiedza każde miasto dokładnie raz i wraca do miasta początkowego?* Parafrazując to na język teorii grafów, zadanie polega na znalezieniu minimalnego cyklu Hamiltona w grafie ważonym.

Istnieje wiele pochodnych tego problemu w zależności od tego jaki dokładnie problem staramy się rozwiązać. W tym projekcie wziąłem pod uwagę **grafy pełne i nieskierowane**. Oznacza to, że każdy węzeł grafu jest połączony ze wszystkimi innymi węzłami, a krawędzie między nimi mają te same wagi w obie strony (tj. dla danych węzłów A i B A -> B == B -> A ).

## 2. Rozwiązanie problemu

### 2.1 Sposób kodowania grafów

Grafy zostały zakodowane jako dwuwymiarowa tablica połączeń. Węzły oznaczone są numerami od 0 do n-1, gdzie n to ilość wezłów w całym grafie. Do generowania losowych grafów zaimplementowałem funkcję *generate_adjmatrix*, która przyjmuję liczbę całkowitą określającą liczbę węzłów i generuje odpowiednią tablicę dwuwymiarową. Wagi krawędzi są liczbami pseudolosowymi z zakresu <1, 10>.


```python
    import random
    def generate_adjmatrix(n, prob):
        m = [[0] * n for i in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if prob < random.random():
                    m[i][j] = m[j][i] = 0
                else:
                    m[i][j] = m[j][i] = random.randint(1, 10)
        return m
    adjmatrix = generate_adjmatrix(5, 1)
    for row in adjmatrix:
        print(''.join([str(n).rjust(3, ' ') for n in row]))
```

      0  4 10  5  6
      4  0  7  7  8
     10  7  0  8  7
      5  7  8  0  1
      6  8  7  1  0
    

### 2.2 Algorytmy inspirowane biologicznie

#### 2.2.1 Algorytm Genetyczny

Pierwszą metodą rozwiązania TSP jest algorytm genetyczny. Chromosomy zakodowane są jako lista zawierające wszystkie węzły grafu - [0..n-1]. Funkcja fitness iteruje przez wygenerowane rozwiązanie, sprawdzając przy tym, czy jakiś węzeł nie pojawił się ponownie. Napisane zostały dwie implementacje tej funkcji - pierwsza sprawdza, czy pierwszy element jest węzłem 0 - w przeciwnym wypadku rozwiązanie jest uznawane jako nieważnie i niwelowane - druga natomiast pozwala pierwszemu elementowi być dowolnym węzłem. Z racji tego, że algorytm genetyczny wyznacza maksimum, za każdy pokonany węzęł od wyniku końcowego odejmowana jest jego waga; w ten sposób algorytm dochodzić będzie do jak najkrótszej trasy.



```python
   def fitness_func(solution, solution_idx):
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

        return fitness - adjmatrix[solution[-1]][solution[0]
```


```python
   def fitness_func2(solution, solution_idx):
        fitness = 0
        allnodes = list(range(len(adjmatrix)))
        max_weight = 10

        # Dowolność początkowego węzła !
        
        for n in range(0, len(solution) - 1):

            # powtórzony node - konczy działanie funkcji
            # zwraca dotychczasowy fitness i karę na podstawie pozostałych wezlow

            if solution[n] not in allnodes:
                return fitness - len(allnodes)*(max_weight * 10)

            fitness -= adjmatrix[solution[n]][solution[n + 1]]
            allnodes.remove(solution[n])

        if solution[-1] not in allnodes:
            return fitness - len(allnodes)*(max_weight * 10)

        return fitness - adjmatrix[solution[-1]][solution[0]
```

Przed porównaniem tego algorytmu z innymi algorytmami postanowiłem przeprowadzić testy różnych wariantów algorytmu ( tj. różnych rodzajów argumentów kontrolujących algorytmem ), na które pozwala biblioteka pygad. Wszystkie testy były wykonane na tym samym grafie o n = 20 i powtórzone 10 razy, aby zebrać uśrednione wyniki i w pewnym stopniu zmiejszyć wpływ elementów losowych. Kolorem czerwonym oznaczone są próby, w których najlepsze wyniki nie spełniały nawet warunków postawionych w problemie - nie przechodziły przez wszystkie wierzchołki tylko 1 raz.

![tabela.png](attachment:tabela.png)

#### 2.2.2 Algorytm kolonii mrówek

Algorytm ten, inspirowany prawdziwym zachowaniem mrówek w przyrodzie, jest często stosowanym algorytmem do rozwiązywania problemów opartych na grafie. "Mrówki" stawiane są na wybranym węźle grafu i poruszają się po nim, zostawiając feromony. Intensywność feromonu na danej ścieżce zależy oczywiście od ilości podążających nią mrówek oraz od tego, jak dobra jest droga, która dana mrówka podąża. Wybór następnego ruchu jest losowo wybierany z dostępnych węzłów, każdy jednak ma inna szansę na bycie wybranym w zależności od jakości ścieżki i ilości feromonów.


```python
def ACO(d, iteration, n_ants, e, alpha, beta):
    n_cities = len(d)
    m = n_ants
    n = n_cities

    # calculating the visibility of the next city visibility(i,j)=1/d(i,j)
    visibility = 1 / d
    visibility[visibility == inf] = 0

    # intializing pheromone present at the paths to the cities
    pheromone = .1 * np.ones((m, n))

    # intializing the route of the ants with size route(n_ants,n_cities+1)
    # we're adding 1 because we want to come back to the fisrt city
    route = np.ones((m, n + 1))

    for ite in range(iteration):

        route[:, 0] = 1  # initial starting and ending positon of every ants '1' i.e city '1'

        for i in range(m):

            temp_visibility = np.array(visibility)  # creating a copy of visibility

            for j in range(n - 1):
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
                
                r = np.random.random_sample()  # random float in [0,1)

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

    print()
    def f(x):
        return x-1
    print('best path :', f(best_route))
    print('cost of the best path', int(dist_min_cost[0]) + d[int(best_route[-2]) - 1, 0])
```

Funkcja ta przyjmuje kilka argumentów:
- *d* - tablica dwuwymiarowa zawierająca dane grafu,
- *iteration* - liczba iteracji algorytmu
- *n_ants* - liczba agentów algorytmu, generujących ścieżki i zostawiających feromony
- *e* - liczba zmiennoprzecinkowa z przedziału (0, 1), określająca wyparowanie feromonów przy następnej iteracji
- *alpha* - liczba całkowita, alpha > 0, wykorzystywana przy obliczaniu wartości feromonu
- *beta* - liczba całkowita, beta > 0, wykorzystywana przy obliczaniu jakości ścieżki

### 2.3 Pozostałe algorytmy

#### 2.3.1 Nearest neighbour algorithm ( algorytm najbliższego sąsiada )

Funkcja ta przyjmuje kilka argumentów:

d - tablica dwuwymiarowa zawierająca dane grafu,
iteration - liczba iteracji algorytmu
n_ants - liczba agentów algorytmu, generujących ścieżki i zostawiających feromony
e - liczba zmiennoprzecinkowa z przedziału (0, 1), określająca wyparowanie feromonów przy następnej iteracji
alpha - liczba całkowita, alpha > 0, wykorzystywana przy obliczaniu wartości feromonu
beta - liczba całkowita, beta > 0, wykorzystywana przy obliczaniu jakości ścieżki

### 2.3 Pozostałe algorytmy
#### 2.3.1 Nearest neighbour algorithm ( algorytm najbliższego sąsiada )

Jest to algorytm typu zachłannego, który opiera się na bardzo prostej zasadzie: w każdym następnym kroku wybierany jest węzeł, który nie został jeszcze odwiedzony i do którego prowadzi najkrótsza ścieżka spośród dostępnych ścieżek. Złożoność czasowa algorytmu wynosi O(n^2). Algorytm ten zwykle nie daje optymalnego rozwiązania, jednak w testowanych przeze mnie przypadkach wypadł nie najgorzej. Poniżej implementacja algorytmu NN:



```python
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
```

#### 2.3.2 Algorytm Helda-Karpa
Algorytm Helda-Karpa rozwiązuje problem TSP w oparciu o programowanie dynamiczne. Zaczynając od miasta 1, algorytm oblicza dla każdego zbioru miast S = {2..n} i dla każdego miasta e różnego od 1 niezawierającego się w S, najkrótszą ścieżkę od 1 do e taką, że przechodzi ona przez wszystkie miasta należące do S w pewnej kolejności. Mówiąc obrazowo, musimy za każdym razem ustalać, który punkt powinien być przedostatni na trasie (który punkt ma poprzedzać punkt e). Na końcu wyznacza się rozwiązanie całego problemu. W tym celu należy znaleźć poprzednika punktu 1 korzystając ze wzoru: minx∈{2, …, n}( D({2, …, n}, x) + dx,1).

Algorytm Helda-Karpa ma złożoność czasową rzędu O(n^2 \* 2^n); jest ona znacznie lepsza niż w przypadku podejścia brute-force, gdzie mamy złożoność rzędu O(n!). Z drugiej strony jego złożoność pamięciowa wynosi O(n \* 2^n), a więc  przy 16GB pamięci systemowej nie byłem w stanie obliczyć rozwiązania dla grafu o rozmiarze ledwie 25 węzłów.


```python
def held_karp(dists):

    n = len(dists)

    # Maps each subset of the nodes to the cost to reach that subset, as well
    # as what node it passed before reaching this subset.
    # Node subsets are represented as set bits.
    C = {}

    # Set transition cost from initial state
    for k in range(1, n):
        C[(1 << k, k)] = (dists[0][k], 0)

    # Iterate subsets of increasing length and store intermediate results
    # in classic dynamic programming manner
    for subset_size in range(2, n):
        for subset in itertools.combinations(range(1, n), subset_size):
            # Set bits for all nodes in this subset
            bits = 0
            for bit in subset:
                bits |= 1 << bit

            # Find the lowest cost to get to this subset
            for k in subset:
                prev = bits & ~(1 << k)

                res = []
                for m in subset:
                    if m == 0 or m == k:
                        continue
                    res.append((C[(prev, m)][0] + dists[m][k], m))
                C[(bits, k)] = min(res)

    # We're interested in all bits but the least significant (the start state)
    bits = (2**n - 1) - 1

    # Calculate optimal cost
    res = []
    for k in range(1, n):
        res.append((C[(bits, k)][0] + dists[k][0], k))
    opt, parent = min(res)

    # Backtrack to find full path
    path = []
    for i in range(n - 1):
        path.append(parent)
        new_bits = bits & ~(1 << parent)
        _, parent = C[(bits, parent)]
        bits = new_bits

    # Add implicit start state
    path.append(0)

    return opt, list(reversed(path))
```

## 3. Ewaulacja rozwiązań

Testy zostały przeprowadzone po 5 razy dla grafów o rozmiarach 15, 20, 25 i 30. Algorytm Helda-Karpa był uwzględniony tylko w testach dla grafów o rozmiarze 15 i 20 ze względu na złożoność pamięciową.

Argumenty algorytmu generycznego:
- sol_per_pop = 50
- num_parents_mating = 25
- num_generations = 3000
- keep_parents = 4
- parent_selection_type = "sss"
- crossover_type = "single_point"
- mutation_type = "random"
- mutation_percent_genes = 6

Argumenty algorytmu ACO:
- iterations = 300
- n_ants = 50
- e = 0.5
- alpha = 1
- beta = 2

![n15.png](attachment:n15.png)

![n20.png](attachment:n20.png)

![n25.png](attachment:n25.png)

![n30.png](attachment:n30.png)

![time.png](attachment:time.png)

Algorytm genetyczny spisał się zdecydowanie najgorzej - rezultaty z niego otrzymane okazały się być nawet gorsze od prostego algorytmu NN. Może to wynikać ze zbyt surowej implementacji funkcji fitness lub słabego dopasowania algorytmu do problemu. Zwiększenie liczby iteracji może do pewnego stopnia rozwiązać ten problem kosztem czasu. Z drugiej strony optymalizacja kolonią mrówek bardzo dobrze poradziła sobie z tym problemem, co nie powinno być zaskoczeniem. Wraz ze wzrostem liczby węzłów otrzymywany został na relatywnie podobnym poziomie, gdzie przy algorytmie genetycznym można zauważyć znaczny wzrost.
