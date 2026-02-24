import pandas as pd
import matplotlib.pyplot as plt
import project1
import numpy as np

# df = pd.read_csv('rrnn_results.csv')

# # plot 1: k vs median cost
# k_data = df[df['experiment'] == 'k']
# plt.figure()
# plt.plot(k_data['value'], k_data['median_cost'], marker='o')
# plt.xlabel('k')
# plt.ylabel('Median Cost')
# plt.title('RRNN: k vs Median Cost')
# plt.savefig('k_plot.png')
# plt.show()

# # plot 2: num_repeats vs median cost
# repeats_data = df[df['experiment'] == 'num_repeats']
# plt.figure()
# plt.plot(repeats_data['value'], repeats_data['median_cost'], marker='o')
# plt.xlabel('num_repeats')
# plt.ylabel('Median Cost')
# plt.title('RRNN: num_repeats vs Median Cost')
# plt.savefig('num_repeats_plot.png')
# plt.show()

# --- NN, NN2OPT, RRNN2OPT PLOTS --- 

# df = pd.read_csv('algorithm_results.csv')
# algorithms = df['algorithm'].unique()
# colors = {'nearest_neighbor': 'blue', 'nearest_neighbor_2opt': 'orange', 'rrnn_2opt': 'green'}

# # plot 1: runtime
# plt.figure()
# for alg in algorithms:
#     data = df[df['algorithm'] == alg]
#     plt.plot(data['n_cities'], data['median_runtime'], marker='o', label=alg, color=colors[alg])
# plt.xlabel('Number of Cities')
# plt.ylabel('Median Runtime (ns)')
# plt.title('Runtime vs Number of Cities')
# plt.xticks([5, 10, 15, 20, 25, 30])
# plt.legend()
# plt.savefig('runtime_plot.png')
# plt.show()

# # plot 2: cpu time
# plt.figure()
# for alg in algorithms:
#     data = df[df['algorithm'] == alg]
#     plt.plot(data['n_cities'], data['median_cpu_time'], marker='o', label=alg, color=colors[alg])
# plt.xlabel('Number of Cities')
# plt.ylabel('Median CPU Time (ns)')
# plt.title('CPU Time vs Number of Cities')
# plt.xticks([5, 10, 15, 20, 25, 30])
# plt.legend()
# plt.savefig('cpu_time_plot.png')
# plt.show()

# # plot 3: cost
# plt.figure()
# for alg in algorithms:
#     data = df[df['algorithm'] == alg]
#     plt.plot(data['n_cities'], data['median_cost'], marker='o', label=alg, color=colors[alg])
# plt.xlabel('Number of Cities')
# plt.ylabel('Median Cost')
# plt.title('Cost vs Number of Cities')
# plt.xticks([5, 10, 15, 20, 25, 30])
# plt.legend()
# plt.savefig('cost_plot.png')
# plt.show()

# --- HILL CLIMBING PLOTS --- 

# df = pd.read_csv('hill_climbing.csv')

# plt.figure()
# plt.plot(df['value'], df['median_cost'], marker='o')
# plt.xlabel('Number of Restarts')
# plt.ylabel('Median Cost')
# plt.title('Hill Climbing: Number of Restarts vs Median Cost')
# plt.savefig('hill_climbing.png')
# plt.show()

# --- ASTAR PLOTS --- 

# df = pd.read_csv('astar_results.csv')

# plt.figure()
# plt.plot(df['n_cities'], df['median_nodes_expanded'], marker='o', color='red')
# plt.xlabel('Number of Cities')
# plt.ylabel('Median Nodes Expanded')
# plt.title('A*: Nodes Expanded vs Number of Cities')
# plt.xticks(df['n_cities'])
# plt.savefig('astar_nodes_expanded.png')
# plt.show()

# --- ASTAR PLOTS COMPARING TO NN, NN2OPT, RRNN2OPT --- 

# astar_df = pd.read_csv('astar_results.csv')
# alg_df = pd.read_csv('algorithm_results_compare_astar.csv')

# # only keep sizes that a* ran on
# valid_sizes = astar_df['n_cities'].tolist()
# alg_df = alg_df[alg_df['n_cities'].isin(valid_sizes)]

# algorithms = ['nearest_neighbor', 'nearest_neighbor_2opt', 'rrnn_2opt']
# colors = {'nearest_neighbor': 'blue', 'nearest_neighbor_2opt': 'orange', 'rrnn_2opt': 'green'}

# metrics = [
#     ('median_runtime', 'Runtime', 'runtime_comparison.png'),
#     ('median_cpu_time', 'CPU Time', 'cpu_comparison.png'),
#     ('median_cost', 'Cost', 'cost_comparison.png'),
# ]

# for metric, label, filename in metrics:
#     plt.figure()
#     for alg in algorithms:
#         alg_data = alg_df[alg_df['algorithm'] == alg].set_index('n_cities')
#         astar_data = astar_df.set_index('n_cities')

#         ratios = alg_data[metric] / astar_data[metric]

#         plt.plot(ratios.index, ratios.values, marker='o', label=alg, color=colors[alg])

#     plt.xlabel('Number of Cities')
#     plt.ylabel(f'{label} / A* {label}')
#     plt.title(f'{label} Relative to A*')
#     plt.xticks(valid_sizes)
#     plt.legend()
#     plt.savefig(filename)
#     plt.show()

# --- HILL CLIMBING COSTS OVER ITERATION --- 

# mat = np.loadtxt('matrices/25_random_adj_mat_0.txt')

# _, _, history = project1.hill_climbing(mat, num_restarts=500)

# plt.figure()
# plt.plot(range(len(history)), history)
# plt.xlabel('Restart')
# plt.ylabel('Best Cost Found')
# plt.title('Hill Climbing Cost Over Restarts')
# plt.savefig('simulated_annealing_iterations')
# plt.show()

# --- SIMULATED ANNEALING COSTS OVER ITERATION --- 

# mat = np.loadtxt('matrices/25_random_adj_mat_0.txt')

# _, _, history = project1.simulated_annealing(mat, 0.995, 1000, 50)

# plt.figure()
# plt.plot(range(len(history)), history)
# plt.xlabel('Iterations')
# plt.ylabel('Best Cost Found')
# plt.title('Simulated Annealing Cost Over Restarts')
# plt.savefig('simulated_annealing_iterations')
# plt.show()

# --- GENETIC ALGORITHM COSTS OVER ITERATION --- 

# mat = np.loadtxt('matrices/25_random_adj_mat_0.txt')

# _, _, history = project1.genetic(mat, 0.1, 50, 200)

# plt.figure()
# plt.plot(range(len(history)), history)
# plt.xlabel('Generations')
# plt.ylabel('Best Cost Found')
# plt.title('Genetic Algorithm Cost Over Restarts')
# plt.savefig('genetic_iterations')
# plt.show()

# --- HC, SA, GA HYPERPARAMETER PLOTS --- 

# df = pd.read_csv('hyperparameter_results.csv')

# # plot 1: hill climbing num_restarts change
# hill_climbing = df[df['algorithm'] == 'hill_climbing']
# plt.figure()
# plt.plot(hill_climbing['value'], hill_climbing['median_cost'], marker='o')
# plt.xlabel('Number of Restarts')
# plt.ylabel('Median Cost')
# plt.title('Hill Climbing: Number of Restarts')
# plt.savefig('hill_climbing_hyperparameter.png')
# plt.show()

# # plot 2: simulated annealing alpha change
# simulated_annealing = df[df['algorithm'] == 'simulated_annealing']
# plt.figure()
# plt.plot(simulated_annealing['value'], simulated_annealing['median_cost'], marker='o')
# plt.xlabel('Alpha')
# plt.ylabel('Median Cost')
# plt.title('Simulated Annealing: Change in Alpha')
# plt.savefig('simulated_annealing_hyperparameter.png')
# plt.show()

# # plot 1: simulated annealing alpha change
# genetic = df[df['algorithm'] == 'genetic']
# plt.figure()
# plt.plot(genetic['value'], genetic['median_cost'], marker='o')
# plt.xlabel('Population Size')
# plt.ylabel('Median Cost')
# plt.title('Genetic: Population Size')
# plt.savefig('genetic_hyperparameter.png')
# plt.show()

# --- COMPARE HC, SA, GA TO ASTAR --- 


astar_df = pd.read_csv('astar_results.csv')
alg_df = pd.read_csv('hc_sa_ga_compare_astar.csv')

# only keep sizes that a* ran on
valid_sizes = astar_df['n_cities'].tolist()
alg_df = alg_df[alg_df['n_cities'].isin(valid_sizes)]

algorithms = ['hill_climbing', 'simulated_annealing', 'genetic']
colors = {'hill_climbing': 'blue', 'simulated_annealing': 'orange', 'genetic': 'green'}

metrics = [
    ('median_runtime', 'Runtime', 'hc_sa_ga_runtime_comparison.png'),
    ('median_cpu_time', 'CPU Time', 'hc_sa_ga_cpu_comparison.png'),
    ('median_cost', 'Cost', 'hc_sa_ga_cost_comparison.png'),
]

for metric, label, filename in metrics:
    plt.figure()
    for alg in algorithms:
        alg_data = alg_df[alg_df['algorithm'] == alg].set_index('n_cities')
        astar_data = astar_df.set_index('n_cities')

        ratios = alg_data[metric] / astar_data[metric]

        plt.plot(ratios.index, ratios.values, marker='o', label=alg, color=colors[alg])

    plt.xlabel('Number of Cities')
    plt.ylabel(f'{label} / A* {label}')
    plt.title(f'{label} Relative to A*')
    plt.xticks(valid_sizes)
    plt.legend()
    plt.savefig(filename)
    plt.show()