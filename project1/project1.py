import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import random
import math
import time
from search import astar_search, Problem
from scipy.sparse.csgraph import minimum_spanning_tree



class TSPProblem(Problem):
    def __init__(self, matrix):
        self.matrix = matrix
        self.cities = list(range(len(matrix)))
        self.nodes_expanded = 0
        super().__init__((0, (0,))) # initialized with starting city 0, and 0 visited cities (current, visited)

    def actions(self, state):
        curr_city, visited = state
        possible_actions = []

        if len(visited) == len(self.cities):
            return [0]

        for city in self.cities:
            if city not in visited and city != 0:
                possible_actions.append(city)
        
        return possible_actions

    def result(self, state, action):
        curr_city, visited = state
        return (action, visited + (action,))

    def goal_test(self, state):
        curr_city, visited = state
        return curr_city == 0 and len(visited) > 1

    def path_cost(self, c, state1, action, state2):
        curr_city, _ = state1
        return c + self.matrix[curr_city][action]

    def h(self, node):
        self.nodes_expanded += 1
        curr_city, visited = node.state
        unvisited = [c for c in range(len(self.cities)) if c not in visited]

        if not unvisited:
            return 0
        
        cities = [curr_city] + unvisited

        submatrix = self.matrix[np.ix_(cities, cities)]
        
        # compute MST and sum its edges
        mst = minimum_spanning_tree(submatrix)
        return mst.toarray().sum()

    
def nearest_neighbor(matrix, start_city=0):
    unseen = set(range(len(matrix))) # e.g. (0,1,2,3)
    curr = start_city
    unseen.remove(curr) 
    tour = []
    tour.append(curr)
    total_dist = 0

    while unseen:
        closest_city = None
        min_dist = float('inf')

        for city in unseen:
            if matrix[curr][city] < min_dist:
                closest_city = city
                min_dist = matrix[curr][city]
            
        tour.append(closest_city)
        total_dist += min_dist
        curr = closest_city
        unseen.remove(curr)

    total_dist += matrix[curr][start_city]
    tour.append(start_city)

    

    return tour, total_dist

def nearest_neighbor_2opt(tour, matrix, start_city=0):
    tour = tour[:-1] # gets rid of the last city (duplicate of the first)

    improved = True
    while improved: # only restart if we found an improvement
        improved = False

        for i in range(1, len(tour) - 1):
            for j in range(i + 1, len(tour)):
                new_j = (j + 1) % len(tour) # account for potential wraparound

                curr_dist = matrix[tour[j]][tour[new_j]] + matrix[tour[i - 1]][tour[i]]
                new_dist = matrix[tour[i-1]][tour[j]] + matrix[tour[i]][tour[new_j]]

                if new_dist < curr_dist:
                    tour[i:j+1] = reversed(tour[i:j+1])
                    improved = True
                    break
            if improved:
                break

    new_tour = tour + [tour[0]]

    final_dist = 0
    for i in range(len(new_tour) - 1):
        final_dist += matrix[new_tour[i]][new_tour[i + 1]]
    
    return new_tour, final_dist
    
def rrnn_2opt(matrix, start_city=0, k=3, num_repeats=5):
    best_tour = None
    best_dist = float('inf')

    k = min(k, len(matrix) - 1)

    for i in range(num_repeats):
        unseen = set(range(len(matrix)))
        curr = start_city
        unseen.remove(curr)
        tour = [curr]

        while unseen:
            distances = []
            for city in unseen:
                distances.append((city, matrix[curr, city])) # tuple of (city index, distance to city from curr)
            distances.sort(key=lambda x: x[1]) # sorts by distance ascending

            k_closest = distances[:k]

            potential_cities = []
            for city, dist in k_closest:
                potential_cities.append(city)

            next_city = random.choice(potential_cities)

            tour.append(next_city)
            unseen.remove(next_city)
            curr = next_city
        tour.append(start_city)

        tour, dist = nearest_neighbor_2opt(tour, matrix, start_city)

        if dist < best_dist:
            best_tour = tour
            best_dist = dist
    
    return best_tour, best_dist    

# helper function to find the distance of a tour
def find_dist(matrix, tour):
    dist = 0
    for i in range(len(tour) - 1):
        dist += matrix[tour[i]][tour[i + 1]]

    return dist

def hill_climbing(matrix, num_restarts):
    best_dist = float('inf')
    best_tour = None

    for i in range(num_restarts): 
        cities = list(range(len(matrix)))
        random.shuffle(cities) # randomize starting tour
        tour = cities + [cities[0]] # adding the start node to the end

        curr_dist = find_dist(matrix, tour)
        
        improved = True
        while improved:
            improved = False
            # get 2 random city indices
            possible_city_swaps = list(range(len(matrix) - 1))
            first_city_index = random.choice(possible_city_swaps)
            possible_city_swaps.remove(first_city_index)
            second_city_index = random.choice(possible_city_swaps)

            # swap the two random cities
            tour[first_city_index], tour[second_city_index] = tour[second_city_index], tour[first_city_index]

            new_dist = find_dist(matrix, tour)

            if new_dist < curr_dist:
                curr_dist = new_dist
                improved = True
            else:
                tour[first_city_index], tour[second_city_index] = tour[second_city_index], tour[first_city_index]
    
        if curr_dist < best_dist:
            best_tour = tuple(tour)
            best_dist = curr_dist
    
    return list(best_tour), best_dist
    
def simulated_annealing(matrix, alpha, initial_temperature, max_iterations):
    best_dist = float('inf')
    best_tour = None
    curr_temperature = initial_temperature
    cities = list(range(len(matrix)))
    random.shuffle(cities) # randomize starting tour
    tour = cities + [cities[0]] # adding the start node to the end

    curr_dist = find_dist(matrix, tour)

    for i in range(max_iterations): 
        if curr_temperature <= 0:
            break

        # get 2 random city indices
        possible_city_swaps = list(range(len(matrix) - 1))
        first_city_index = random.choice(possible_city_swaps)
        possible_city_swaps.remove(first_city_index)
        second_city_index = random.choice(possible_city_swaps)

        # swap the two random cities
        tour[first_city_index], tour[second_city_index] = tour[second_city_index], tour[first_city_index]

        new_dist = find_dist(matrix, tour)

        accept_worse = math.exp((curr_dist - new_dist) / curr_temperature)

        if new_dist < curr_dist or random.random() < accept_worse:
            curr_dist = new_dist
            curr_temperature = curr_temperature * alpha
        else:
            tour[first_city_index], tour[second_city_index] = tour[second_city_index], tour[first_city_index]

        if curr_dist < best_dist:
            best_tour = tuple(tour)
            best_dist = curr_dist
    
    return list(best_tour), best_dist
    
# offspring function for genetic
def create_offspring(parent1, parent2):
    p1 = parent1[:-1]
    p2 = parent2[:-1]
    n = len(p1)

    start = random.randint(0, n - 1)
    end = random.randint(start + 1, n)
    segment = p1[start:end]

    remaining = []
    for city in p2:
        if city not in segment:
            remaining.append(city)

    child = remaining[:start] + segment + remaining[start:]
    return child + [child[0]]

def genetic(matrix, mutation_chance, population_size, num_generations):
    population = []
    cities = list(range(len(matrix)))
    # create random starting parents
    for i in range(population_size):
        random_cities = cities.copy()
        random.shuffle(random_cities)
        random_cities += [random_cities[0]] # add start to end as well
        population.append(random_cities)
    
    for i in range(num_generations):
        # pair up parents
        for i in range(population_size):
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            while parent2 == parent1:
                parent2 = random.choice(population)

            child = create_offspring(parent1, parent2)

            if random.random() < mutation_chance: # mutation occurs
                possible_city_swaps = list(range(1, len(matrix) - 1))
                first_city_index = random.choice(possible_city_swaps)
                possible_city_swaps.remove(first_city_index)
                second_city_index = random.choice(possible_city_swaps)

                child[first_city_index], child[second_city_index] = child[second_city_index], child[first_city_index]
            
            population.append(child)
        
        # sort in place
        population.sort(key=lambda tour: find_dist(matrix, tour))
        
        population = population[:population_size] # keep only the elites
    
    # find best of the best
    best_tour = population[0]
    best_dist = find_dist(matrix, best_tour)
    return best_tour, best_dist

def astar(matrix):
    problem = TSPProblem(matrix)
    solution = astar_search(problem)

    return solution.solution(), solution.path_cost

def main():
    mat = np.loadtxt(sys.argv[1])

    start_time = time.time()
    start_cpu = time.process_time()

    tour, dist = hill_climbing(mat, 10)

    end_time = time.time()
    end_cpu = time.process_time()

    print(f"Tour: {tour}")
    print(f"Cost: {dist}")
    print(f"Runtime: {end_time - start_time:.4f} seconds")
    print(f"CPU Time: {end_cpu - start_cpu:.4f} seconds")

if __name__ == '__main__':
    main()