import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set larger font sizes for readability in a paper
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11
})

# Read data from file
data_file = 'gpu_algos.txt'
df = pd.read_csv(data_file, sep=r'\s+', engine='python', comment='//')

# Clean up matrix names by removing .mtx extension
df['Matrix'] = df['Matrix'].str.replace('.mtx', '')

# Get unique algorithms and matrices
algorithms = df['Algorithm'].unique()
matrices = df['Matrix'].unique()

# Function to get short algorithm names
def get_short_name(algo_name):
    return algo_name.split('-')[-1]

# Set width of bars
bar_width = 0.2
index = np.arange(len(matrices))

# Colors for different algorithms - use distinct colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Create first figure for Bandwidth
fig1, ax1 = plt.subplots(figsize=(11, 6))  # Wider figure to accommodate legend

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
    bars1 = ax1.bar(position, bandwidths, bar_width, 
                   label=get_short_name(algo), color=colors[i % len(colors)])

# Add labels and title to bandwidth plot
ax1.set_xlabel('Matrix', fontsize=14)
ax1.set_ylabel('Bandwidth (GB/s)', fontsize=14)
ax1.set_title('Memory Bandwidth by Algorithm and Matrix', fontsize=16)
ax1.set_xticks(index)
ax1.set_xticklabels(matrices, rotation=45, ha='right')
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Linear scale (no logarithmic scale)
# Position legend outside the plot to the right
ax1.legend(title="Algorithm", loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True)

# Add a note about configuration
note_text = "All GPU algorithms using 1024 threads, 64 blocks where applicable"
plt.figtext(0.5, 0.01, note_text, ha='center', fontsize=10, style='italic')

# Adjust layout to make room for the legend
plt.tight_layout(rect=[0, 0.05, 0.85, 0.95])

# Save bandwidth plot
bandwidth_output = 'gpu_bandwidth_comparison.png'
plt.savefig(bandwidth_output, dpi=300, bbox_inches='tight')
print(f"Bandwidth plot saved as {bandwidth_output}")

# Create second figure for GFLOPS
fig2, ax2 = plt.subplots(figsize=(11, 6))  # Wider figure to accommodate legend

# Plot GFLOPS
for i, algo in enumerate(algorithms):
    algo_data = df[df['Algorithm'] == algo]
    # Create a dictionary to ensure matrices are in the right order
    gflops_dict = {matrix: 0 for matrix in matrices}
    
    for _, row in algo_data.iterrows():
        gflops_dict[row['Matrix']] = row['GFLOPS']
    
    # Get GFLOPS values in the same order as matrices list
    gflops_values = [gflops_dict[matrix] for matrix in matrices]
    
    # Plot with an offset to create grouped bars
    position = index + (i - len(algorithms)/2 + 0.5) * bar_width
    bars2 = ax2.bar(position, gflops_values, bar_width, 
                   label=get_short_name(algo), color=colors[i % len(colors)])

# Add labels and title to GFLOPS plot
ax2.set_xlabel('Matrix', fontsize=14)
ax2.set_ylabel('Performance (GFLOPS)', fontsize=14)
ax2.set_title('Computational Performance by Algorithm and Matrix', fontsize=16)
ax2.set_xticks(index)
ax2.set_xticklabels(matrices, rotation=45, ha='right')
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# Linear scale (no logarithmic scale)
# Position legend outside the plot to the right
ax2.legend(title="Algorithm", loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True)

# Add the same configuration note
plt.figtext(0.5, 0.01, note_text, ha='center', fontsize=10, style='italic')

# Adjust layout to make room for the legend
plt.tight_layout(rect=[0, 0.05, 0.85, 0.95])

# Save GFLOPS plot
gflops_output = 'gpu_gflops_comparison.png'
plt.savefig(gflops_output, dpi=300, bbox_inches='tight')
print(f"GFLOPS plot saved as {gflops_output}")

# Show the plots
plt.show()