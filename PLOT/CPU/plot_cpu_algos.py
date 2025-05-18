import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl

# Set up a professional paper-ready style
plt.style.use('seaborn-v0_8-paper')

# Configure fonts for publication quality with smaller font sizes
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 9,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'legend.title_fontsize': 9,
    'figure.dpi': 300,
})

# Read data from file
data_file = 'cpu_algos.txt'
df = pd.read_csv(data_file, sep=r'\s+', engine='python', comment='//')

# Clean up matrix names by removing .mtx extension
df['Matrix'] = df['Matrix'].str.replace('.mtx', '')

# Get unique algorithms and matrices
algorithms = sorted(df['Algorithm'].unique(), reverse=True)  # Sort in reverse to get naive first
matrices = df['Matrix'].unique()

# Function to get short algorithm names
def get_short_name(algo_name):
    return algo_name.split('-')[-1]

# Create a figure with two subplots side by side (more compact)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
fig.subplots_adjust(wspace=0.25)  # Reduce space between plots

# Set width of bars
bar_width = 0.3  # Slightly wider than GPU plots since fewer algorithms
index = np.arange(len(matrices))

# Professional color palette with good contrast for print
colors = ['#d73027', '#4575b4']  # Red for naive, Blue for multipleAcc (swapped order)

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
                   label=get_short_name(algo), 
                   color=colors[i % len(colors)],
                   edgecolor='black', linewidth=0.5)

# Add labels and title to bandwidth plot - no x-axis label
ax1.set_ylabel('Bandwidth (GB/s)', fontweight='bold')
ax1.set_title('Bandwidth', fontweight='bold', pad=10)
ax1.set_xticks(index)
ax1.set_xticklabels(matrices, rotation=45, ha='right', fontsize=7)
ax1.grid(axis='y', linestyle='--', alpha=0.3, color='gray')

# Add subtle spines
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_linewidth(0.5)
ax1.spines['bottom'].set_linewidth(0.5)

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
    bars2 = ax2.bar(position, gflops, bar_width, 
                   label=get_short_name(algo), 
                   color=colors[i % len(colors)],
                   edgecolor='black', linewidth=0.5)

# Add labels and title to GFLOPS plot - no x-axis label
ax2.set_ylabel('GFLOPS', fontweight='bold')
ax2.set_title('GFLOPS', fontweight='bold', pad=10)
ax2.set_xticks(index)
ax2.set_xticklabels(matrices, rotation=45, ha='right', fontsize=7)
ax2.grid(axis='y', linestyle='--', alpha=0.3, color='gray')

# Add subtle spines
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_linewidth(0.5)
ax2.spines['bottom'].set_linewidth(0.5)

# Add a single legend for both plots at the top of the figure
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, 
          title="Algorithm", 
          loc='upper center', 
          bbox_to_anchor=(0.5, 0.98),
          ncol=len(algorithms),  # Side by side for the algorithms
          frameon=True,
          framealpha=0.9,
          edgecolor='lightgray')

# Add a main title - make it smaller and more subtle
fig.suptitle('CPU SpMV Performance Comparison', 
             fontsize=12, y=1.02, fontweight='bold')

# Adjust layout to make room for titles and legend
plt.tight_layout(rect=[0, 0, 1, 0.93])

# Save the figure in high resolution
output_file = 'cpu_performance_comparison.png'
plt.savefig(output_file, dpi=600, bbox_inches='tight')
print(f"Plot saved as {output_file}")

# Show the plot
plt.show()