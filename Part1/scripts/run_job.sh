#!/bin/bash
# filepath: c:\Users\gille\Desktop\GPU-Computing-2025-257820\Part1\scripts\run_job.sh

# Get the parameters
EXECUTABLE="$1"
MATRIX="$2"
ADDITIONAL_ARGS="$3"
EXEC_NAME="$4"

# Extract matrix name for output
MATRIX_NAME=$(basename "$MATRIX")

# Check if executable exists
if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: Executable does not exist: $EXECUTABLE"
    exit 1
fi

# Check if matrix exists
if [ ! -f "$MATRIX" ]; then
    echo "Error: Matrix file does not exist: $MATRIX"
    exit 1
fi

# Create the SBATCH command
SBATCH_CMD="sbatch"
SBATCH_CMD+=" --job-name=${EXEC_NAME}_${MATRIX_NAME}"
SBATCH_CMD+=" --output=outputs/${EXEC_NAME}_${MATRIX_NAME}_%j.txt"
SBATCH_CMD+=" --partition=edu-short"
SBATCH_CMD+=" --nodes=1"
SBATCH_CMD+=" --ntasks-per-node=1"
SBATCH_CMD+=" --cpus-per-task=1"

# Add GPU resources - specify gpu:0 for CPU jobs, gpu:1 for GPU jobs
if [[ "$EXECUTABLE" == *"GPU"* ]]; then
    SBATCH_CMD+=" --gres=gpu:1"
    # Wrap the commands in a script
    SBATCH_CMD+=" --wrap=\"module load CUDA/12.1.1 && echo 'Running on matrix: $MATRIX_NAME' && $EXECUTABLE $MATRIX $ADDITIONAL_ARGS\""
else
    # CPU job - explicitly specify gpu:0
    SBATCH_CMD+=" --gres=gpu:0"
    SBATCH_CMD+=" --wrap=\"echo 'Running on matrix: $MATRIX_NAME' && $EXECUTABLE $MATRIX $ADDITIONAL_ARGS\""
fi

# Print and execute the command
echo -e "\nSubmitting job with the following command:"
echo "$EXECUTABLE $MATRIX $ADDITIONAL_ARGS"
echo "Using SBATCH command: $SBATCH_CMD"

# Execute the command and capture the job ID
JOB_SUBMISSION=$(eval $SBATCH_CMD)
JOB_ID=$(echo $JOB_SUBMISSION | grep -oP 'Submitted batch job \K[0-9]+')

echo "Job submitted with ID: $JOB_ID"
echo "Output will be in: outputs/${EXEC_NAME}_${MATRIX_NAME}_${JOB_ID}.txt"