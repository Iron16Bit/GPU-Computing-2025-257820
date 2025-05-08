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
	@mkdir -p "$(CPU_BIN_FOLDER)"
	$(CC) $< -o $@ $(TIME_LIB_OBJ) -I$(INC_FOLDER) $(OPT) -lm

# Compile each GPU binary
$(GPU_BIN_FOLDER)/%.exec: $(GPU_FOLDER)/%.cu
	@mkdir -p "$(GPU_BIN_FOLDER)"
	$(NVCC) $(GPU_ARCH) -m64 -o $@ $<

# Compile timing library object
$(TIME_LIB_OBJ): $(SRC_FOLDER)/my_time_lib.c
	@mkdir -p "$(BIN_FOLDER)" "$(OBJ_FOLDER)" "$(BATCH_OUT_FOLDER)"
	$(CC) -c $< -o $@ -I$(INC_FOLDER) $(OPT)

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
