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

df = pd.read_csv('algorithm_results.csv')
algorithms = df['algorithm'].unique()
colors = {'nearest_neighbor': 'blue', 'nearest_neighbor_2opt': 'orange', 'rrnn_2opt': 'green'}

# plot 1: runtime
plt.figure()
for alg in algorithms:
    data = df[df['algorithm'] == alg]
    plt.plot(data['n_cities'], data['median_runtime'], marker='o', label=alg, color=colors[alg])
plt.xlabel('Number of Cities')
plt.ylabel('Median Runtime (ns)')
plt.title('Runtime vs Number of Cities')
plt.xticks([5, 10, 15, 20, 25, 30])
plt.legend()
plt.savefig('runtime_plot.png')
plt.show()

# plot 2: cpu time
plt.figure()
for alg in algorithms:
    data = df[df['algorithm'] == alg]
    plt.plot(data['n_cities'], data['median_cpu_time'], marker='o', label=alg, color=colors[alg])
plt.xlabel('Number of Cities')
plt.ylabel('Median CPU Time (ns)')
plt.title('CPU Time vs Number of Cities')
plt.xticks([5, 10, 15, 20, 25, 30])
plt.legend()
plt.savefig('cpu_time_plot.png')
plt.show()

# plot 3: cost
plt.figure()
for alg in algorithms:
    data = df[df['algorithm'] == alg]
    plt.plot(data['n_cities'], data['median_cost'], marker='o', label=alg, color=colors[alg])
plt.xlabel('Number of Cities')
plt.ylabel('Median Cost')
plt.title('Cost vs Number of Cities')
plt.xticks([5, 10, 15, 20, 25, 30])
plt.legend()
plt.savefig('cost_plot.png')
plt.show()