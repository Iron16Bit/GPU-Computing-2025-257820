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
DATA_FOLDER := Data
SCRIPTS_FOLDER := scripts

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

# Default target checks for data, loads modules, then builds all
all: check_data load_modules build_all

# Check if Data folder exists and download/sort data if needed
check_data:
	@if [ ! -d "$(DATA_FOLDER)" ] || [ -z "$$(ls -A $(DATA_FOLDER) 2>/dev/null)" ]; then \
        echo "Data folder missing or empty. Downloading and processing data..."; \
        bash $(SCRIPTS_FOLDER)/download_data.sh; \
        bash $(SCRIPTS_FOLDER)/sort_files.sh; \
    else \
        echo "Data folder exists and contains files. Skipping download."; \
    fi

# Build all CPU and GPU programs
build_all: $(CPU_BINS) $(GPU_BINS)

# Compile each CPU binary with the time lib
$(CPU_BIN_FOLDER)/%: $(CPU_FOLDER)/%.c $(TIME_LIB_OBJ)
	@mkdir -p "$(CPU_BIN_FOLDER)"
	$(CC) $< -o $@ $(TIME_LIB_OBJ) -I$(INC_FOLDER) $(OPT) -lm

# Compile each GPU binary
$(GPU_BIN_FOLDER)/%.exec: $(GPU_FOLDER)/%.cu
	@mkdir -p "$(GPU_BIN_FOLDER)"
	module load CUDA/12.1.1 && $(NVCC) $(GPU_ARCH) -m64 -o $@ $<

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

# Mark phony targets that don't create files
.PHONY: all check_data load_modules build_all clean clean_batch_outputs