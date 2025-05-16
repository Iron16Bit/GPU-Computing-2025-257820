#!/bin/bash

# Create outputs directory if it doesn't exist
mkdir -p outputs

# Arrays of values to test
THREADS_PER_BLOCK=(32 128 256 512 1024)
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