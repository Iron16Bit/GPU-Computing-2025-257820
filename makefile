# Compiler and flags
CC = gcc
NVCC = nvcc
OPT = -std=c99 -O3
GPU_ARCH = --gpu-architecture=sm_80

# Folder structure
SRC_FOLDER := TIMER_LIB/src
INC_FOLDER := TIMER_LIB/include
OBJ_FOLDER := TIMER_LIB/obj
BIN_FOLDER := bin
CPU_FOLDER := CPU
GPU_FOLDER := GPU
CPU_BIN_FOLDER := $(BIN_FOLDER)/CPU
GPU_BIN_FOLDER := $(BIN_FOLDER)/GPU
BATCH_OUT_FOLDER := outputs
SLURM_SCRIPTS_FOLDER := slurm_scripts

# Timing library object
TIME_LIB_OBJ := $(OBJ_FOLDER)/my_time_lib.o

# Find all .c files in CPU_FOLDER
CPU_SRCS := $(wildcard $(CPU_FOLDER)/*.c)
# Create a matching list of CPU binaries
CPU_BINS := $(patsubst $(CPU_FOLDER)/%.c, $(CPU_BIN_FOLDER)/%, $(CPU_SRCS))

# Find all .cu files in GPU_FOLDER
GPU_SRCS := $(wildcard $(GPU_FOLDER)/*.cu)
# Create a matching list of GPU binaries
GPU_BINS := $(patsubst $(GPU_FOLDER)/%.cu, $(GPU_BIN_FOLDER)/%.exec, $(GPU_SRCS))

# Default target builds all CPU and GPU programs
all: $(CPU_BINS) $(GPU_BINS)

# Compile each CPU binary with the time lib
$(CPU_BIN_FOLDER)/%: $(CPU_FOLDER)/%.c $(TIME_LIB_OBJ)
    @mkdir -p $(CPU_BIN_FOLDER)
    $(CC) $< -o $@ $(TIME_LIB_OBJ) -I$(INC_FOLDER) $(OPT) -lm

# Compile each GPU binary
$(GPU_BIN_FOLDER)/%.exec: $(GPU_FOLDER)/%.cu
    @mkdir -p $(GPU_BIN_FOLDER)
    $(NVCC) $(GPU_ARCH) -m64 -o $@ $<

# Compile timing library object
$(TIME_LIB_OBJ): $(SRC_FOLDER)/my_time_lib.c
    @mkdir -p $(BIN_FOLDER) $(OBJ_FOLDER) $(BATCH_OUT_FOLDER)
    $(CC) -c $< -o $@ -I$(INC_FOLDER) $(OPT)

# Run all CPU binaries with a given input path and log output to log.txt
run-cpu:
    @echo "Usage: make run-cpu PATH=your/input/file" > log.txt
    @if [ -z "$(PATH)" ]; then \
        echo "Error: PATH not specified." >> log.txt; \
        exit 1; \
    fi
    @for bin in $(CPU_BINS); do \
        echo "Running $$bin with input $(PATH)" >> log.txt; \
        $$bin $(PATH) >> log.txt 2>&1; \
        echo "----" >> log.txt; \
    done
    @echo "All programs finished. Output saved to log.txt" >> log.txt

# Create SLURM job script for GPU runs
$(SLURM_SCRIPTS_FOLDER)/run_gpu.sh:
    @mkdir -p $(SLURM_SCRIPTS_FOLDER)
    @echo "#!/bin/bash" > $@
    @echo "#SBATCH --job-name=run_exec" >> $@
    @echo "#SBATCH --partition=edu-short" >> $@
    @echo "#SBATCH --nodes=1" >> $@
    @echo "#SBATCH --gres=gpu:1" >> $@
    @echo "#SBATCH --ntasks-per-node=1" >> $@
    @echo "#SBATCH --cpus-per-task=1" >> $@
    @echo "" >> $@
    @echo "# Get the name of the executable from the first argument" >> $@
    @echo "EXECUTABLE=\"\$$1\"" >> $@
    @echo "" >> $@
    @echo "if [ -z \"\$$EXECUTABLE\" ]; then" >> $@
    @echo "    echo \"Error: No executable provided.\"" >> $@
    @echo "    echo \"Usage: sbatch run_exec.sh <executable> [args...]\"" >> $@
    @echo "    exit 1" >> $@
    @echo "fi" >> $@
    @echo "" >> $@
    @echo "# Optional arguments" >> $@
    @echo "shift" >> $@
    @echo "ARGS=\"\$$@\"" >> $@
    @echo "" >> $@
    @echo "# Get the hostname" >> $@
    @echo "HOSTNAME=\$$(hostname)" >> $@
    @echo "" >> $@
    @echo "./\"\$$EXECUTABLE\" \"\$$HOSTNAME\" \$$ARGS" >> $@
    @chmod +x $@

# Run all GPU binaries with SLURM with a given input path
run-gpu: $(SLURM_SCRIPTS_FOLDER)/run_gpu.sh
    @echo "Submitting GPU jobs to SLURM..."
    @if [ -z "$(PATH)" ]; then \
        echo "Error: PATH not specified."; \
        echo "Usage: make run-gpu PATH=your/input/file"; \
        exit 1; \
    fi
    @for bin in $(GPU_BINS); do \
        echo "Submitting SLURM job for $$bin with input $(PATH)"; \
        sbatch $(SLURM_SCRIPTS_FOLDER)/run_gpu.sh $$bin $(PATH); \
    done
    @echo "All GPU jobs submitted to SLURM. Check status with 'squeue'"

# Clean outputs
clean_batch_outputs:
    rm -f $(BATCH_OUT_FOLDER)/*

# Clean binaries and objects
clean:
    rm -rf $(BIN_FOLDER)
    rm -rf $(OBJ_FOLDER)
    
# Clean SLURM scripts
clean-slurm:
    rm -rf $(SLURM_SCRIPTS_FOLDER)