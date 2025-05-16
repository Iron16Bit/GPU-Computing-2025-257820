import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set larger font sizes for readability
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11
})

# Read data from file
data_file = 'cpu_algos.txt'
df = pd.read_csv(data_file, sep=r'\s+', engine='python', comment='//')

# Clean up matrix names by removing .mtx extension
df['Matrix'] = df['Matrix'].str.replace('.mtx', '')

# Get unique algorithms and matrices
algorithms = df['Algorithm'].unique()
matrices = df['Matrix'].unique()

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.subplots_adjust(wspace=0.3)  # Add some space between plots

# Set width of bars
bar_width = 0.35
index = np.arange(len(matrices))

# Colors for different algorithms
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# Plot Bandwidth
for i, algo in enumerate(algorithms):
    algo_data = df[df['Algorithm'] == algo]
    # Create a dictionary to ensure matrices are in the right order
    bandwidth_dict = {matrix: 0 for matrix in matrices}
    
    for _, row in algo_data.iterrows():
        bandwidth_dict[row['Matrix']] = row['Bandwidth']
    
    # Get bandwidth values in the same order as matrices list
    bandwidths = [bandwidth_dict[matrix] for matrix in matrices]
    
    # Plot with an offset to create grouped bars
    position = index + (i - len(algorithms)/2 + 0.5) * bar_width
    bars1 = ax1.bar(position, bandwidths, bar_width, label=algo.split('-')[-1], color=colors[i % len(colors)])

# Add labels and title to bandwidth plot
ax1.set_xlabel('Matrix')
ax1.set_ylabel('Bandwidth (GB/s)')
ax1.set_title('Memory Bandwidth by Algorithm and Matrix')
ax1.set_xticks(index)
ax1.set_xticklabels(matrices, rotation=45, ha='right')
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Plot GFLOPS
for i, algo in enumerate(algorithms):
    algo_data = df[df['Algorithm'] == algo]
    # Create a dictionary to ensure matrices are in the right order
    gflops_dict = {matrix: 0 for matrix in matrices}
    
    for _, row in algo_data.iterrows():
        gflops_dict[row['Matrix']] = row['GFLOPS']
    
    # Get GFLOPS values in the same order as matrices list
    gflops = [gflops_dict[matrix] for matrix in matrices]
    
    # Plot with an offset to create grouped bars
    position = index + (i - len(algorithms)/2 + 0.5) * bar_width
    bars2 = ax2.bar(position, gflops, bar_width, label=algo.split('-')[-1], color=colors[i % len(colors)])

# Add labels and title to GFLOPS plot
ax2.set_xlabel('Matrix')
ax2.set_ylabel('Performance (GFLOPS)')
ax2.set_title('Computational Performance by Algorithm and Matrix')
ax2.set_xticks(index)
ax2.set_xticklabels(matrices, rotation=45, ha='right')
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# Add legend - just put it on the second subplot to save space
ax2.legend(title="Algorithm", bbox_to_anchor=(1.05, 1), loc='upper left')

# Add a main title
fig.suptitle('CPU SpMV Performance Comparison', fontsize=16, y=0.98)

# Adjust layout to make room for titles and labels
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save the figure in high resolution
output_file = 'cpu_performance_comparison.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Plot saved as {output_file}")

# Show the plot
plt.show()