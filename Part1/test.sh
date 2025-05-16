#!/bin/bash

# Create outputs directory if it doesn't exist
mkdir -p outputs

# Arrays of values to test
THREADS_PER_BLOCK=(32 128 256 512 1024 2048)
BLOCKS=(1 4 8 16 32 64 128)

# Define GPU and CPU matrices based on the original names
GPU_MATRICES=(
    # Diagonal matrices for GPU
    "CurlCurl_4"
    "af_shell8"
    "af23560"
    # Unstructured matrices for GPU
    "mawi_201512012345"
    "human_gene2"
    "lhr10"
)

CPU_MATRICES=(
    # Unstructured matrices for CPU
    "pesa"
    "lhr02"
    "bcsstk08"
    # Diagonal matrices for CPU
    "cz5108"
    "Chebyshev3"
    "DK01R"
)

# Find all executables in bin directory
echo "Finding all executables..."
CPU_EXECUTABLES=()
GPU_EXECUTABLES=()

# List CPU executables
for exec in bin/CPU/*; do
    if [ -x "$exec" ]; then
        CPU_EXECUTABLES+=("$exec")
    fi
done

# List GPU executables
for exec in bin/GPU/*.exec; do
    if [ -x "$exec" ]; then
        GPU_EXECUTABLES+=("$exec")
    fi
done

echo "Found ${#CPU_EXECUTABLES[@]} CPU executables and ${#GPU_EXECUTABLES[@]} GPU executables"
echo "Starting batch runs..."

# Function to run a single job
run_job() {
    local exec="$1"
    local matrix="$2"
    local args="$3"
    local name="$4"
    
    echo "Running: $exec $matrix $args"
    bash scripts/run_job.sh "$exec" "$matrix" "$args" "$name"
    
    # Small delay to avoid overwhelming scheduler
    sleep 1
}

# Process CPU executables with CPU matrices
echo "Processing CPU executables..."
for exec in "${CPU_EXECUTABLES[@]}"; do
    exec_name=$(basename "$exec")
    echo "Processing executable: $exec_name"
    
    for matrix_base in "${CPU_MATRICES[@]}"; do
        # Find the actual matrix file (might have _sorted.mtx suffix)
        matrix_file=$(find Data/ -name "${matrix_base}*.mtx" | head -1)
        
        if [ -n "$matrix_file" ] && [ -f "$matrix_file" ]; then
            matrix_name=$(basename "$matrix_file")
            echo "  With matrix: $matrix_name"
            run_job "$exec" "$matrix_file" "" "${exec_name}"
        else
            echo "  Warning: Matrix file for $matrix_base not found"
        fi
    done
done

# Process GPU executables with GPU matrices
echo "Processing GPU executables..."
for exec in "${GPU_EXECUTABLES[@]}"; do
    exec_name=$(basename "$exec")
    # Remove .exec extension if present
    exec_name="${exec_name%.exec}"
    echo "Processing executable: $exec_name"
    
    for matrix_base in "${GPU_MATRICES[@]}"; do
        # Find the actual matrix file (might have _sorted.mtx suffix)
        matrix_file=$(find Data/ -name "${matrix_base}*.mtx" | head -1)
        
        if [ -n "$matrix_file" ] && [ -f "$matrix_file" ]; then
            matrix_name=$(basename "$matrix_file")
            echo "  With matrix: $matrix_name"
            
            # Different handling based on executable type
            if [[ "$exec_name" == "SpMV-GPU-rowSplit" || "$exec_name" == "SpMV-GPU-tpv" ]]; then
                # Only test threads per block for these executables
                for tpb in "${THREADS_PER_BLOCK[@]}"; do
                    args="$tpb"
                    run_job "$exec" "$matrix_file" "$args" "${exec_name}_tpb${tpb}"
                done
            elif [[ "$exec_name" == "SpMV-GPU-sequential" || "$exec_name" == "SpMV-GPU-stride" ]]; then
                # Test both threads per block and number of blocks
                for tpb in "${THREADS_PER_BLOCK[@]}"; do
                    for blk in "${BLOCKS[@]}"; do
                        args="$tpb $blk"
                        run_job "$exec" "$matrix_file" "$args" "${exec_name}_tpb${tpb}_b${blk}"
                    done
                done
            else
                # Other GPU executables, run with no parameters
                run_job "$exec" "$matrix_file" "" "$exec_name"
            fi
        else
            echo "  Warning: Matrix file for $matrix_base not found"
        fi
    done
done

echo "All batch jobs submitted. Check 'squeue' for job status."
echo "Results will be in the outputs directory."

# Wait for all jobs to complete
read -p "Do you want to wait for all jobs to complete (y/n)? " wait_response
if [[ "$wait_response" == "y" || "$wait_response" == "Y" ]]; then
    echo "Waiting for jobs to complete. This may take some time..."
    sleep 10  # Initial delay
    
    # Check job count every 30 seconds
    while true; do
        job_count=$(squeue -u $USER | wc -l)
        # Subtract 1 for the header line
        job_count=$((job_count - 1))
        
        if [ $job_count -eq 0 ]; then
            echo "All jobs completed!"
            break
        else
            echo "$job_count jobs still running. Checking again in 30 seconds..."
            sleep 30
        fi
    done
    
    # Create a summary report
    echo "Creating performance summary..."
    echo "Performance Summary" > performance_summary.txt
    echo "===================" >> performance_summary.txt
    echo "Generated at: $(date)" >> performance_summary.txt
    echo "" >> performance_summary.txt
    
    echo "CPU Algorithms" >> performance_summary.txt
    echo "=============" >> performance_summary.txt
    for exec_name in $(ls outputs/ | grep -v "GPU" | cut -d '_' -f 1 | sort -u); do
        echo "Executable: $exec_name" >> performance_summary.txt
        echo "------------------------" >> performance_summary.txt
        
        # Extract performance data
        grep "Average \|elapsed time" outputs/${exec_name}_* | sort -n -k 4 | head -n 10 >> performance_summary.txt
        echo "" >> performance_summary.txt
    done
    
    echo "GPU Algorithms" >> performance_summary.txt
    echo "=============" >> performance_summary.txt
    for exec_name in $(ls outputs/ | grep "GPU" | cut -d '_' -f 1 | sort -u); do
        echo "Executable: $exec_name" >> performance_summary.txt
        echo "------------------------" >> performance_summary.txt
        
        # Extract performance data
        grep "Average \|elapsed time" outputs/${exec_name}_* | sort -n -k 4 | head -n 10 >> performance_summary.txt
        echo "" >> performance_summary.txt
    done
    
    echo "Summary saved to performance_summary.txt"
fi