import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import random
from search import astar_search, Problem
from scipy.sparse.csgraph import minimum_spanning_tree



class TSPProblem(Problem):
    def __init__(self, matrix):
        self.matrix = matrix
        self.cities = list(range(len(matrix)))
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



def main():
    mat = np.loadtxt('matrices/15_random_adj_mat_0.txt')

    print(mat)

    # for start in range(len(mat)):
    #     tour, dist = nearest_neighbor(mat, start)
    #     print(f"{tour}, Regular Distance: {dist}")
    #     tour, dist = nearest_neighbor_2opt(tour, mat, start)
    #     print(f"{tour}, 2-Opt Distance: {dist}")
    #     tour, dist = rrnn_2opt(mat, start)
    #     print(f"{tour}, RRNN2-Opt Distance: {dist}")
    problem = TSPProblem(mat)
    solution = astar_search(problem)
    if solution:
        print(f"A* tour: {solution.solution()}, Distance: {solution.path_cost}")


if __name__ == '__main__':
    main()