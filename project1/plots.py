import pandas as pd
import matplotlib.pyplot as plt

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

astar_df = pd.read_csv('astar_results.csv')
alg_df = pd.read_csv('algorithm_results_compare_astar.csv')

# only keep sizes that a* ran on
valid_sizes = astar_df['n_cities'].tolist()
alg_df = alg_df[alg_df['n_cities'].isin(valid_sizes)]

algorithms = ['nearest_neighbor', 'nearest_neighbor_2opt', 'rrnn_2opt']
colors = {'nearest_neighbor': 'blue', 'nearest_neighbor_2opt': 'orange', 'rrnn_2opt': 'green'}

metrics = [
    ('median_runtime', 'Runtime', 'runtime_comparison.png'),
    ('median_cpu_time', 'CPU Time', 'cpu_comparison.png'),
    ('median_cost', 'Cost', 'cost_comparison.png'),
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