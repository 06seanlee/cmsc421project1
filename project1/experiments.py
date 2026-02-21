import csv
import time
import numpy as np
import project1

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

sizes = [5, 10, 15, 20, 25, 30]
matrices = []
for size in sizes:
    for i in range(10):
        matrices.append(f'matrices/{size}_random_adj_mat_{i}.txt')


algorithms = [
    ('nearest_neighbor', lambda m: project1.nearest_neighbor(m), 100),
    ('nearest_neighbor_2opt', lambda m: project1.nearest_neighbor_2opt(project1.nearest_neighbor(m)[0], m), 100),
    ('rrnn_2opt', lambda m: project1.rrnn_2opt(m, k=3, num_repeats=5), 1)
]

results = []

for name, algorithm, num_runs in algorithms:
    for size in sizes:
        runtimes = []
        cpu_times = []
        costs = []

        # get all 10 matrices for this size
        size_matrices = [f for f in matrices if f.startswith(f'matrices/{size}_')]

        for matrix_file in size_matrices:
            mat = np.loadtxt(matrix_file)

            start_time = time.time_ns()
            start_cpu = time.process_time_ns()
            for _ in range(num_runs):
                tour, dist = algorithm(mat)
            end_time = time.time_ns()
            end_cpu = time.process_time_ns()

            runtime = (end_time - start_time) / num_runs
            cpu_time = (end_cpu - start_cpu) / num_runs

            runtimes.append(runtime)
            cpu_times.append(cpu_time)
            costs.append(dist)

        results.append({
            'algorithm': name,
            'n_cities': size,
            'median_runtime': np.median(runtimes),
            'median_cpu_time': np.median(cpu_times),
            'median_cost': np.median(costs)
        })

with open('algorithm_results.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['algorithm', 'n_cities', 'median_runtime', 'median_cpu_time', 'median_cost'])
    writer.writeheader()
    writer.writerows(results)