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
    'xtick.labelsize': 8,  # Smaller font for matrix names
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'legend.title_fontsize': 9,
    'figure.dpi': 300,
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

# Set width of bars (slightly thinner for a cleaner look)
bar_width = 0.18
index = np.arange(len(matrices))

# Professional color palette with good contrast for print
colors = ['#4575b4', '#d73027', '#91bfdb', '#fc8d59']  # Blue, Red, Light blue, Light red

# Create first figure for Bandwidth with higher quality settings
fig1, ax1 = plt.subplots(figsize=(5.5, 4.0))  # Standard figure size for papers

# Plot Bandwidth - without hatching patterns
for i, algo in enumerate(algorithms):
    algo_data = df[df['Algorithm'] == algo]
    bandwidth_dict = {matrix: 0 for matrix in matrices}
    
    for _, row in algo_data.iterrows():
        bandwidth_dict[row['Matrix']] = row['Bandwidth']
    
    bandwidths = [bandwidth_dict[matrix] for matrix in matrices]
    
    position = index + (i - len(algorithms)/2 + 0.5) * bar_width
    bars1 = ax1.bar(position, bandwidths, bar_width, 
                    label=get_short_name(algo), 
                    color=colors[i % len(colors)],
                    edgecolor='black', linewidth=0.5)

# Add labels and title with professional formatting
# Remove x-axis label (Matrix)
ax1.set_ylabel('Bandwidth (GB/s)', fontweight='bold')
ax1.set_title('Memory Bandwidth Performance', fontweight='bold', pad=10)
ax1.set_xticks(index)
ax1.set_xticklabels(matrices, rotation=45, ha='right', fontsize=7)  # Even smaller font for matrix names
ax1.grid(axis='y', linestyle='--', alpha=0.3, color='gray')

# Add subtle spines
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_linewidth(0.5)
ax1.spines['bottom'].set_linewidth(0.5)

# Position legend OUTSIDE the plot area to the right
# This is the key change to prevent overlap
ax1.legend(title="Algorithm", 
           loc='center left',
           bbox_to_anchor=(1.01, 0.5),  # Position to the right of the plot
           frameon=True, 
           ncol=1,  # Vertical stack to save horizontal space
           framealpha=0.9,
           edgecolor='lightgray')

# Add a note about configuration - more subtly
note_text = "1024 threads, 64 blocks"
plt.figtext(0.5, 0.01, note_text, ha='center', fontsize=7, style='italic', color='dimgray')

# Use tight layout with adjusted margins
plt.tight_layout(rect=[0, 0.03, 0.85, 0.98])  # Reserve space on right for legend

# Save bandwidth plot in publication quality
bandwidth_output = 'gpu_bandwidth_comparison.png'
plt.savefig(bandwidth_output, dpi=600, bbox_inches='tight')
plt.savefig(bandwidth_output.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
print(f"Bandwidth plot saved as {bandwidth_output}")

# Create second figure for GFLOPS
fig2, ax2 = plt.subplots(figsize=(5.5, 4.0))

# Plot GFLOPS with same style for consistency
for i, algo in enumerate(algorithms):
    algo_data = df[df['Algorithm'] == algo]
    gflops_dict = {matrix: 0 for matrix in matrices}
    
    for _, row in algo_data.iterrows():
        gflops_dict[row['Matrix']] = row['GFLOPS']
    
    gflops_values = [gflops_dict[matrix] for matrix in matrices]
    
    position = index + (i - len(algorithms)/2 + 0.5) * bar_width
    bars2 = ax2.bar(position, gflops_values, bar_width, 
                    label=get_short_name(algo), 
                    color=colors[i % len(colors)],
                    edgecolor='black', linewidth=0.5)

# Add labels and title with professional formatting
# Remove x-axis label (Matrix)
ax2.set_ylabel('Performance (GFLOPS)', fontweight='bold')
ax2.set_title('Computational Performance', fontweight='bold', pad=10)
ax2.set_xticks(index)
ax2.set_xticklabels(matrices, rotation=45, ha='right', fontsize=7)  # Even smaller font for matrix names
ax2.grid(axis='y', linestyle='--', alpha=0.3, color='gray')

# Add subtle spines
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_linewidth(0.5)
ax2.spines['bottom'].set_linewidth(0.5)

# Match the legend style - also position outside the plot
ax2.legend(title="Algorithm", 
           loc='center left', 
           bbox_to_anchor=(1.01, 0.5),  # Position to the right of the plot
           frameon=True, 
           ncol=1,  # Vertical stack
           framealpha=0.9,
           edgecolor='lightgray')

# Add the same configuration note
plt.figtext(0.5, 0.01, note_text, ha='center', fontsize=7, style='italic', color='dimgray')

# Use tight layout with adjusted margins
plt.tight_layout(rect=[0, 0.03, 0.85, 0.98])  # Reserve space on right for legend

# Save GFLOPS plot in publication quality
gflops_output = 'gpu_gflops_comparison.png'
plt.savefig(gflops_output, dpi=600, bbox_inches='tight')
plt.savefig(gflops_output.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
print(f"GFLOPS plot saved as {gflops_output}")

# Show the plots
plt.show()