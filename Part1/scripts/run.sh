#!/bin/bash
# filepath: c:\Users\gille\Desktop\GPU-Computing-2025-257820\Part1\scripts\run_gpu_job.sh

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
    if [[ "$exec_name" == "SpMV-GPU-rowSplit.exec" || "$exec_name" == "SpMV-GPU-tpv.exec" ]]; then
        read -p "Enter threads per block (default: 256): " threads_per_block
        if [ -n "$threads_per_block" ]; then
            additional_args="$threads_per_block"
        fi
    elif [[ "$exec_name" == "SpMV-GPU-sequential.exec" || "$exec_name" == "SpMV-GPU-stride.exec" ]]; then
        read -p "Enter threads per block (default: 256): " threads_per_block
        read -p "Enter number of blocks (default: 4): " num_blocks
        
        if [ -n "$threads_per_block" ] && [ -n "$num_blocks" ]; then
            additional_args="$threads_per_block $num_blocks"
        elif [ -n "$threads_per_block" ]; then
            additional_args="$threads_per_block"
        fi
    fi
fi

# Create SLURM job script
job_script="slurm_job_${exec_name}_$(date +%s).sh"

cat > "$job_script" << EOF
#!/bin/bash
#SBATCH --job-name=${exec_name}
#SBATCH --output=outputs/${exec_name}_%j.out
#SBATCH --partition=edu-short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
EOF

# Add GPU resource request for GPU executables
if [[ "$selected_exec" == *"GPU"* ]]; then
    echo "#SBATCH --gres=gpu:1" >> "$job_script"
    echo "module load CUDA/12.1.1" >> "$job_script"
fi

# Add the command to run
echo "" >> "$job_script"
echo "$selected_exec $selected_matrix $additional_args" >> "$job_script"

# Make executable and submit
chmod +x "$job_script"
echo -e "\nSubmitting job with the following command:"
echo "$selected_exec $selected_matrix $additional_args"
echo "Submitting job script: $job_script"

sbatch "$job_script"
echo "Job submitted. Check outputs directory for results."