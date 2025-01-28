import os
import sys
import math
import pandas as pd
import argparse
import subprocess
import concurrent.futures
import matplotlib.pyplot as plt

results_df = pd.read_csv('results_max_throughputs.csv')

# Filter the DataFrame for each data parallelism value
dp1_df = results_df[results_df['Data parallelism'] == 1]
dp2_df = results_df[results_df['Data parallelism'] == 2]
dp4_df = results_df[results_df['Data parallelism'] == 4]
dp8_df = results_df[results_df['Data parallelism'] == 8]

# Plotting Device Utilization
plt.figure(figsize=(5, 5))


# Define markers for different Pipeline parallelism values
markers = {80: 'o', 40: 's', 20: 'D', 16: '^', 10: 'v', 8: '<', 5: '>', 4: 'p'}

fontsize = 19

# Plot each data parallelism value with different colors and markers
for dp_df, color, label in zip([dp1_df, dp2_df, dp4_df, dp8_df], 
                            ['tab:blue', 'tab:green', 'tab:orange', 'tab:red'], 
                            ['DP = 1', 'DP = 2', 'DP = 4', 'DP = 8']):
    for pp_value, marker in markers.items():
        subset_df = dp_df[dp_df['Pipeline parallelism'] == pp_value]
        if not subset_df.empty:
            throughputs = subset_df['Throughput'] / 1000

            # plt.scatter(subset_df['Device number'], throughputs, color=color, marker=marker, s=60, label=f'{label}')
            plt.scatter(subset_df['Device number'], subset_df['Device utilization'], color=color, marker=marker, s=60, label=f'{label}')
            # plt.plot(subset_df['Device number'], subset_df['Device utilization'], color=color, marker=marker, markersize=6, label=f'{label}')

plt.ylim(0, 1.1)

# Add labels and title
plt.xlabel('Device number', fontsize=fontsize)
plt.ylabel('Throughput ( K Tokens/s )', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
# plt.title('Throughput vs Device number for different Data Parallelism values')

# Add legend
plt.legend(fontsize=fontsize)

# Save the plot to a PDF file
plt.savefig('plot_device_utilization.pdf')
# plt.savefig('plot_3.pdf')

# Close the plot
plt.close()


# results_df = pd.read_csv('results_max_throughputs.csv')
# dp1_df = results_df[results_df['Data parallelism'] == 1]
# dp2_df = results_df[results_df['Data parallelism'] == 2]
# dp4_df = results_df[results_df['Data parallelism'] == 4]
# dp8_df = results_df[results_df['Data parallelism'] == 8]

# Plotting Throughputs
plt.figure(figsize=(5, 5))


# Define markers for different Pipeline parallelism values
markers = {80: 'o', 40: 's', 20: 'D', 16: '^', 10: 'v', 8: '<', 5: '>', 4: 'p'}

fontsize = 19

# Plot each data parallelism value with different colors and markers
for dp_df, color, label in zip([dp1_df, dp2_df, dp4_df, dp8_df], 
                            ['tab:blue', 'tab:green', 'tab:orange', 'tab:red'], 
                            ['DP = 1', 'DP = 2', 'DP = 4', 'DP = 8']):
    for pp_value, marker in markers.items():
        subset_df = dp_df[dp_df['Pipeline parallelism'] == pp_value]
        if not subset_df.empty:
            throughputs = subset_df['Throughput'] / 1000

            plt.scatter(subset_df['Device number'], throughputs, color=color, marker=marker, s=60, label=f'{label}')
            # plt.scatter(subset_df['Device number'], subset_df['Device utilization'], color=color, marker=marker, s=60, label=f'{label}')
            # plt.plot(subset_df['Device number'], subset_df['Device utilization'], color=color, marker=marker, markersize=6, label=f'{label}')

# plt.ylim(0, 1.1)

# Add labels and title
plt.xlabel('Device number', fontsize=fontsize)
plt.ylabel('Throughput ( K Tokens/s )', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
# plt.title('Throughput vs Device number for different Data Parallelism values')

# Add legend
plt.legend(fontsize=fontsize)

# Save the plot to a PDF file
# plt.savefig('plot_device_utilization.pdf')
plt.savefig('plot_pp_80_device_up_to_128_half_column.pdf')

# Close the plot
plt.close()




# Plotting Throughputs full column
plt.figure(figsize=(9, 5))


# Define markers for different Pipeline parallelism values
markers = {80: 'o', 40: 's', 20: 'D', 16: '^', 10: 'v', 8: '<', 5: '>', 4: 'p'}

fontsize = 19

# Plot each data parallelism value with different colors and markers
for dp_df, color, label in zip([dp1_df, dp2_df, dp4_df, dp8_df], 
                            ['tab:blue', 'tab:green', 'tab:orange', 'tab:red'], 
                            ['DP = 1', 'DP = 2', 'DP = 4', 'DP = 8']):
    for pp_value, marker in markers.items():
        subset_df = dp_df[dp_df['Pipeline parallelism'] == pp_value]
        if not subset_df.empty:
            throughputs = subset_df['Throughput'] / 1000

            plt.scatter(subset_df['Device number'], throughputs, color=color, marker=marker, s=60, label=f'{label}')
            # plt.scatter(subset_df['Device number'], subset_df['Device utilization'], color=color, marker=marker, s=60, label=f'{label}')
            # plt.plot(subset_df['Device number'], subset_df['Device utilization'], color=color, marker=marker, markersize=6, label=f'{label}')

# plt.ylim(0, 1.1)

# Add labels and title
plt.xlabel('Device number', fontsize=fontsize)
plt.ylabel('Throughput ( K Tokens/s )', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
# plt.title('Throughput vs Device number for different Data Parallelism values')

# Add legend
plt.legend(fontsize=fontsize)

# Save the plot to a PDF file
# plt.savefig('plot_device_utilization.pdf')
plt.savefig('plot_pp_80_device_up_to_128.pdf')

# Close the plot
plt.close()



    

