import csv
import time
import numpy as np
import project1

# --- FINDING OPTIMAL K AND NUM_REPEATS FOR RRNN2 ---
# sizes = [5, 10, 15, 20, 25, 30]
# matrices = []
# for size in sizes:
#     for i in range(10):
#         matrices.append(f'matrices/{size}_random_adj_mat_{i}.txt')

# results = []

# # experiment #1: vary k, keep num_repeats constant
# k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# fixed_num_repeats = 10

# for k in k_values:
#     costs = []
#     for matrix_file in matrices:
#         mat = np.loadtxt(matrix_file)
#         tour, dist = project1.rrnn_2opt(mat, start_city=0, k=k, num_repeats=fixed_num_repeats)
#         costs.append(dist)
#     results.append({'experiment': 'k', 'value': k, 'median_cost': np.median(costs)})

# # experiment #2: vary num_repeats, keep k constant
# num_repeats_values = [1, 5, 10, 15, 20, 25]
# fixed_k = 3

# for num_repeats in num_repeats_values:
#     costs = []
#     for matrix_file in matrices:
#         mat = np.loadtxt(matrix_file)
#         tour, dist = project1.rrnn_2opt(mat, k=fixed_k, num_repeats=num_repeats)
#         costs.append(dist)
#     results.append({'experiment': 'num_repeats', 'value': num_repeats, 'median_cost': np.median(costs)})

# # save to CSV
# with open('rrnn_results.csv', 'w', newline='') as f:
#     writer = csv.DictWriter(f, fieldnames=['experiment', 'value', 'median_cost'])
#     writer.writeheader()
#     writer.writerows(results)

# --- NN, NN2OPT, RRNN2OPT EXPERIMENTS ---

# sizes = [5, 10, 15, 20, 25, 30]
# matrices = []
# for size in sizes:
#     for i in range(10):
#         matrices.append(f'matrices/{size}_random_adj_mat_{i}.txt')


# algorithms = [
#     ('nearest_neighbor', lambda m: project1.nearest_neighbor(m), 100),
#     ('nearest_neighbor_2opt', lambda m: project1.nearest_neighbor_2opt(project1.nearest_neighbor(m)[0], m), 100),
#     ('rrnn_2opt', lambda m: project1.rrnn_2opt(m, k=3, num_repeats=5), 1)
# ]

# results = []

# for name, algorithm, num_runs in algorithms:
#     for size in sizes:
#         runtimes = []
#         cpu_times = []
#         costs = []

#         # get all 10 matrices for this size
#         size_matrices = [f for f in matrices if f.startswith(f'matrices/{size}_')]

#         for matrix_file in size_matrices:
#             mat = np.loadtxt(matrix_file)

#             start_time = time.time_ns()
#             start_cpu = time.process_time_ns()
#             for _ in range(num_runs):
#                 tour, dist = algorithm(mat)
#             end_time = time.time_ns()
#             end_cpu = time.process_time_ns()

#             runtime = (end_time - start_time) / num_runs
#             cpu_time = (end_cpu - start_cpu) / num_runs

#             runtimes.append(runtime)
#             cpu_times.append(cpu_time)
#             costs.append(dist)

#         results.append({
#             'algorithm': name,
#             'n_cities': size,
#             'median_runtime': np.median(runtimes),
#             'median_cpu_time': np.median(cpu_times),
#             'median_cost': np.median(costs)
#         })

# with open('algorithm_results.csv', 'w', newline='') as f:
#     writer = csv.DictWriter(f, fieldnames=['algorithm', 'n_cities', 'median_runtime', 'median_cpu_time', 'median_cost'])
#     writer.writeheader()
#     writer.writerows(results)

# --- NN, NN2OPT, RRNN2OPT FOR COMPARISON TO A* ---

# sizes = [5,6,7,8,9,10,15]
# matrices = []
# for size in sizes:
#     for i in range(10):
#         matrices.append(f'matrices/{size}_random_adj_mat_{i}.txt')


# algorithms = [
#     ('nearest_neighbor', lambda m: project1.nearest_neighbor(m), 100),
#     ('nearest_neighbor_2opt', lambda m: project1.nearest_neighbor_2opt(project1.nearest_neighbor(m)[0], m), 100),
#     ('rrnn_2opt', lambda m: project1.rrnn_2opt(m, k=3, num_repeats=5), 1)
# ]

# results = []

# for name, algorithm, num_runs in algorithms:
#     for size in sizes:
#         runtimes = []
#         cpu_times = []
#         costs = []

#         # get all 10 matrices for this size
#         size_matrices = [f for f in matrices if f.startswith(f'matrices/{size}_')]

#         for matrix_file in size_matrices:
#             mat = np.loadtxt(matrix_file) 

#             start_time = time.time_ns()
#             start_cpu = time.process_time_ns()
#             for _ in range(num_runs):
#                 tour, dist = algorithm(mat)
#             end_time = time.time_ns()
#             end_cpu = time.process_time_ns()

#             runtime = (end_time - start_time) / num_runs
#             cpu_time = (end_cpu - start_cpu) / num_runs

#             runtimes.append(runtime)
#             cpu_times.append(cpu_time)
#             costs.append(dist)

#         results.append({
#             'algorithm': name,
#             'n_cities': size,
#             'median_runtime': np.median(runtimes),
#             'median_cpu_time': np.median(cpu_times),
#             'median_cost': np.median(costs)
#         })

# with open('algorithm_results_compare_astar.csv', 'w', newline='') as f:
#     writer = csv.DictWriter(f, fieldnames=['algorithm', 'n_cities', 'median_runtime', 'median_cpu_time', 'median_cost'])
#     writer.writeheader()
#     writer.writerows(results)

# --- HILL CLIMBING, SIMULATED ANNEALING, GENETIC EXPERIMENT --- 

# sizes = [5, 10, 15, 20, 25, 30]
# matrices = []
# for size in sizes:
#     for i in range(10):
#         matrices.append(f'matrices/{size}_random_adj_mat_{i}.txt')

# results = []

# num_restarts_values = [1, 5, 10, 15, 20, 25]

# for num_restarts in num_restarts_values:
#     costs = []
#     for matrix_file in matrices:
#         mat = np.loadtxt(matrix_file)
#         tour, dist = project1.hill_climbing(mat, num_restarts)
#         costs.append(dist)
#     results.append({'value': num_restarts, 'median_cost': np.median(costs)})

# with open('hill_climbing.csv', 'w', newline='') as f:
#     writer = csv.DictWriter(f, fieldnames=['value', 'median_cost'])
#     writer.writeheader()
#     writer.writerows(results)

# sizes = [5,6,7,8,9,10,15]

# results = []

# for size in sizes:
#     runtimes = []
#     cpu_times = []
#     costs = []
#     nodes_expanded = []

#     for i in range(10):
#         matrix_file = f'matrices/{size}_random_adj_mat_{i}.txt'
#         mat = np.loadtxt(matrix_file)

#         problem = project1.TSPProblem(mat)

#         start_time = time.time_ns()
#         start_cpu = time.process_time_ns()
#         solution = project1.astar_search(problem)
#         end_time = time.time_ns()
#         end_cpu = time.process_time_ns()

#         if solution:
#             runtimes.append(end_time - start_time)
#             cpu_times.append(end_cpu - start_cpu)
#             costs.append(solution.path_cost)
#             nodes_expanded.append(problem.nodes_expanded)

#     results.append({
#         'n_cities': size,
#         'median_runtime': np.median(runtimes),
#         'median_cpu_time': np.median(cpu_times),
#         'median_cost': np.median(costs),
#         'median_nodes_expanded': np.median(nodes_expanded)
#     })

# with open('astar_results.csv', 'w', newline='') as f:
#     writer = csv.DictWriter(f, fieldnames=['n_cities', 'median_runtime', 'median_cpu_time', 'median_cost', 'median_nodes_expanded'])
#     writer.writeheader()
#     writer.writerows(results)

# --- COST OVER ITERATIONS FOR HC, SA, GA ---

# sizes = [5, 10, 15, 20, 25, 30]
# matrices = []
# for size in sizes:
#     for i in range(10):
#         matrices.append(f'matrices/{size}_random_adj_mat_{i}.txt')

# algorithms = [
#     ('hill_climbing',        lambda m: project1.hill_climbing(m, num_restarts=20)),
#     ('simulated_annealing',  lambda m: project1.simulated_annealing(m, alpha=0.99, initial_temperature=1000, max_iterations=10000)),
#     ('genetic',              lambda m: project1.genetic(m, mutation_chance=0.1, population_size=50, num_generations=200)),
# ]

# --- HC, SA, GA HYPERPARAMETER EXPERIMENT


# results = []

# # use all 10 matrices of size 15 for median cost
# matrices = []
# for i in range(10):
#     matrices.append(np.loadtxt(f'matrices/{15}_random_adj_mat_{i}.txt'))

# # ─────────────────────────────────────────────
# # hill climbing: vary num_restarts
# # ─────────────────────────────────────────────
# print("Hill climbing hyperparameter experiment...")
# for num_restarts in [1, 5, 10, 20, 30, 40, 50]:
#     costs = []
#     for mat in matrices:
#         _, dist, _ = project1.hill_climbing(mat, num_restarts=num_restarts)
#         costs.append(dist)
#     results.append({
#         'algorithm': 'hill_climbing',
#         'hyperparameter': 'num_restarts',
#         'value': num_restarts,
#         'median_cost': np.median(costs)
#     })
#     print(f"  num_restarts={num_restarts} done")

# # ─────────────────────────────────────────────
# # simulated annealing: vary alpha
# # ─────────────────────────────────────────────
# print("Simulated annealing hyperparameter experiment...")
# for alpha in [0.999, 0.99, 0.95, 0.90, 0.85]:
#     costs = []
#     for mat in matrices:
#         _, dist, _ = project1.simulated_annealing(mat, alpha=alpha, initial_temperature=1000, max_iterations=500)
#         costs.append(dist)
#     results.append({
#         'algorithm': 'simulated_annealing',
#         'hyperparameter': 'alpha',
#         'value': alpha,
#         'median_cost': np.median(costs)
#     })
#     print(f"  alpha={alpha} done")

# # ─────────────────────────────────────────────
# # genetic: vary population_size
# # ─────────────────────────────────────────────
# print("Genetic hyperparameter experiment...")
# for population_size in [10, 25, 50, 75, 100]:
#     costs = []
#     for mat in matrices:
#         _, dist, _ = project1.genetic(mat, mutation_chance=0.1, population_size=population_size, num_generations=100)
#         costs.append(dist)
#     results.append({
#         'algorithm': 'genetic',
#         'hyperparameter': 'population_size',
#         'value': population_size,
#         'median_cost': np.median(costs)
#     })
#     print(f"  population_size={population_size} done")

# with open('hyperparameter_results.csv', 'w', newline='') as f:
#     writer = csv.DictWriter(f, fieldnames=['algorithm', 'hyperparameter', 'value', 'median_cost'])
#     writer.writeheader()
#     writer.writerows(results)

# --- COMPARE HC, SA, GC TO ASTAR ---

sizes = [5,6,7,8,9,10,15]
matrices = []
for size in sizes:
    for i in range(10):
        matrices.append(f'matrices/{size}_random_adj_mat_{i}.txt')


algorithms = [
    ('hill_climbing', lambda m: project1.hill_climbing(m, 40)),
    ('simulated_annealing', lambda m: project1.simulated_annealing(m, 0.95, 1000, 300)),
    ('genetic', lambda m: project1.genetic(m, 0.1, 50, 200))
]

results = []

for name, algorithm in algorithms:
    for size in sizes:
        matrix_file = f'matrices/{size}_random_adj_mat_0.txt'
        mat = np.loadtxt(matrix_file)

        start_time = time.time_ns()
        start_cpu = time.process_time_ns()
        tour, dist, _ = algorithm(mat)
        end_time = time.time_ns()
        end_cpu = time.process_time_ns()

        results.append({
            'algorithm': name,
            'n_cities': size,
            'runtime': end_time - start_time,
            'cpu_time': end_cpu - start_cpu,
            'cost': dist
        })

with open('hc_sa_ga_compare_astar.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['algorithm', 'n_cities', 'median_runtime', 'median_cpu_time', 'median_cost'])
    writer.writeheader()
    writer.writerows(results)