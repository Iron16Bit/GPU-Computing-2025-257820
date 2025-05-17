import pandas as pd
import numpy as np

# Read the data file
data_file = 'gpu_algos.txt'
with open(data_file, 'r') as f:
    lines = f.readlines()

# Keep header lines
header_lines = lines[:2]
data_lines = lines[2:]

# Process each line and fix the bandwidth values for SpMV-GPU-rowSplit
corrected_lines = []
for line in data_lines:
    parts = line.strip().split()
    if len(parts) >= 7 and parts[0] == 'SpMV-GPU-rowSplit':
        algorithm = parts[0]
        matrix = parts[1]
        blocks = parts[2]
        tpb = parts[3]
        time_ms = parts[4]
        
        # Multiply bandwidth by 1,000,000 (10^6)
        bandwidth = float(parts[5]) * 1000000
        gflops = parts[6]
        
        # Format the corrected line with proper spacing
        corrected_line = f"{algorithm:<20} {matrix:<30} {blocks:<7} {tpb:<7} {time_ms:<12} {bandwidth:<12.6f} {gflops}\n"
        corrected_lines.append(corrected_line)
    else:
        # Keep other lines unchanged
        corrected_lines.append(line)

# Write the corrected file
with open('gpu_algos_corrected.txt', 'w') as f:
    # Write header
    for line in header_lines:
        f.write(line)
    # Write corrected data lines
    for line in corrected_lines:
        f.write(line)

print("Correction completed. File saved as 'gpu_algos_corrected.txt'")

# Display the original vs corrected values for verification
print("\nOriginal vs Corrected Bandwidth values for SpMV-GPU-rowSplit:")
print("Matrix                       Original    Corrected")
print("---------------------------------------------------")

for line in data_lines:
    parts = line.strip().split()
    if len(parts) >= 7 and parts[0] == 'SpMV-GPU-rowSplit':
        matrix = parts[1]
        original = float(parts[5])
        corrected = original * 1000000
        print(f"{matrix:<30} {original:<10.6f} {corrected:<10.6f}")