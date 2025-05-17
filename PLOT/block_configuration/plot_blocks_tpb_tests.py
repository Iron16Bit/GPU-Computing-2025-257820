import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
import os

# Set larger font sizes for all text elements
plt.rcParams.update({
    'font.size': 14,          # Base font size
    'axes.titlesize': 18,     # Title font size
    'axes.labelsize': 16,     # Axis label font size
    'xtick.labelsize': 14,    # X-tick label size
    'ytick.labelsize': 14,    # Y-tick label size
    'legend.fontsize': 14,    # Legend font size
    'figure.titlesize': 20    # Figure title size
})

# Read the data from the file
data_file = 'blocks_tpb_tests.txt'

# Define column names for the data
column_names = ['Algorithm', 'Matrix', 'Blocks', 'TPB', 'Time(ms)', 'Bandwidth', 'GFLOPS']

# Read the file with specified column names
df = pd.read_csv(data_file, sep=r'\s+', engine='python', names=column_names, 
                 comment='//', skiprows=1)

print(f"Read {len(df)} rows of data")

# Extract the unique blocks and threads per block values
blocks = sorted(df['Blocks'].unique())
tpb = sorted(df['TPB'].unique())

print(f"Found {len(blocks)} block configurations: {blocks}")
print(f"Found {len(tpb)} thread-per-block configurations: {tpb}")

# Create meshgrid for plotting
X, Y = np.meshgrid(blocks, tpb)

# Initialize arrays for bandwidth and GFLOPS
Z_bandwidth = np.zeros((len(tpb), len(blocks)))
Z_gflops = np.zeros((len(tpb), len(blocks)))

# Fill the arrays with data
for i, t in enumerate(tpb):
    for j, b in enumerate(blocks):
        # Get the row with the matching block and TPB values
        row = df[(df['Blocks'] == b) & (df['TPB'] == t)]
        if not row.empty:
            Z_bandwidth[i, j] = row['Bandwidth'].values[0]
            Z_gflops[i, j] = row['GFLOPS'].values[0]

# Create contour plot (2D heatmap) with a more compact layout suitable for 2-column paper
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
plt.subplots_adjust(wspace=0.4)  # Increase space between subplots

# Create log-spaced x-ticks for better visualization
x_ticks = blocks
y_ticks = tpb

# Bandwidth contour plot
c1 = ax1.contourf(np.log2(X), Y, Z_bandwidth, levels=20, cmap='viridis')
ax1.set_title('Bandwidth')
ax1.set_xlabel('Number of Blocks')
ax1.set_ylabel('Threads per Block')
ax1.set_xticks(np.log2(x_ticks))
ax1.set_xticklabels([str(x) for x in x_ticks], rotation=45)  # Rotate labels for better fit
cbar1 = fig.colorbar(c1, ax=ax1)
cbar1.set_label('GB/s')
cbar1.ax.tick_params(labelsize=12)  # Make colorbar ticks a bit smaller

# GFLOPS contour plot
c2 = ax2.contourf(np.log2(X), Y, Z_gflops, levels=20, cmap='plasma')
ax2.set_title('GFLOPS')
ax2.set_xlabel('Number of Blocks')
ax2.set_ylabel('Threads per Block')
ax2.set_xticks(np.log2(x_ticks))
ax2.set_xticklabels([str(x) for x in x_ticks], rotation=45)  # Rotate labels for better fit
cbar2 = fig.colorbar(c2, ax=ax2)
cbar2.set_label('GFLOPS')
cbar2.ax.tick_params(labelsize=12)  # Make colorbar ticks a bit smaller

# Add a main title - make it compact
fig.suptitle('SpMV Performances vs. Block Configuration', y=0.98)

# Make layout tight but with enough padding
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for the title

# Save the contour plot with higher DPI for print quality
contour_output_path = 'blocks_tpb_performances.png'
plt.savefig(contour_output_path, dpi=600, bbox_inches='tight', format='png')
print(f"Heatmap saved to {os.path.abspath(contour_output_path)}")

# Show the plot
plt.show()