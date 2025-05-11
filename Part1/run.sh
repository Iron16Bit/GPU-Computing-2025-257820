#!/bin/bash
# filepath: c:\Users\gille\Desktop\GPU-Computing-2025-257820\Part1\scripts\select_job.sh

# Create outputs directory if it doesn't exist
mkdir -p outputs

# Find all executables in bin directory
echo "Available executables:"
executables=()
index=1

# List CPU executables
for exec in bin/CPU/*; do
    if [ -x "$exec" ]; then
        base_name=$(basename "$exec")
        executables+=("$exec")
        echo "$index) $base_name"
        ((index++))
    fi
done

# List GPU executables
for exec in bin/GPU/*.exec; do
    if [ -x "$exec" ]; then
        base_name=$(basename "$exec" .exec)
        executables+=("$exec")
        echo "$index) $base_name"
        ((index++))
    fi
done

# Get user choice for executable
read -p "Select an executable (1-$((index-1))): " exec_choice
if [ "$exec_choice" -lt 1 ] || [ "$exec_choice" -gt $((index-1)) ]; then
    echo "Invalid selection. Exiting."
    exit 1
fi

selected_exec="${executables[$((exec_choice-1))]}"
exec_name=$(basename "$selected_exec")
# Remove .exec extension if present
exec_name="${exec_name%.exec}"

# Find all matrix files in Data directory
echo -e "\nAvailable matrices:"
matrices=()
index=1

for matrix in Data/*.mtx; do
    if [ -f "$matrix" ]; then
        base_name=$(basename "$matrix")
        matrices+=("$matrix")
        echo "$index) $base_name"
        ((index++))
    fi
done

# Get user choice for matrix
read -p "Select a matrix file (1-$((index-1))): " matrix_choice
if [ "$matrix_choice" -lt 1 ] || [ "$matrix_choice" -gt $((index-1)) ]; then
    echo "Invalid selection. Exiting."
    exit 1
fi

selected_matrix="${matrices[$((matrix_choice-1))]}"

# Parse additional parameters based on executable type
additional_args=""

# Check if the selected executable is GPU-based
if [[ "$selected_exec" == *"GPU"* ]]; then
    if [[ "$exec_name" == "SpMV-GPU-rowSplit" || "$exec_name" == "SpMV-GPU-tpv" ]]; then
        read -p "Enter threads per block (default: 256): " threads_per_block
        if [ -n "$threads_per_block" ]; then
            additional_args="$threads_per_block"
        fi
    elif [[ "$exec_name" == "SpMV-GPU-sequential" || "$exec_name" == "SpMV-GPU-stride" ]]; then
        read -p "Enter threads per block (default: 256): " threads_per_block
        read -p "Enter number of blocks (default: 4): " num_blocks
        
        if [ -n "$threads_per_block" ] && [ -n "$num_blocks" ]; then
            additional_args="$threads_per_block $num_blocks"
        elif [ -n "$threads_per_block" ]; then
            additional_args="$threads_per_block"
        fi
    fi
fi

# Call the run script with all parameters
bash scripts/run_job.sh "$selected_exec" "$selected_matrix" "$additional_args" "$exec_name"