# Compiler and flags
CC = gcc
OPT = -std=c99 -O3

# Folder structure
SRC_FOLDER := TIMER_LIB/src
INC_FOLDER := TIMER_LIB/include
OBJ_FOLDER := TIMER_LIB/obj
BIN_FOLDER := bin
CPU_FOLDER := CPU
CPU_BIN_FOLDER := $(BIN_FOLDER)/CPU
BATCH_OUT_FOLDER := outputs

# Timing library object
TIME_LIB_OBJ := $(OBJ_FOLDER)/my_time_lib.o

# Find all .c files in CPU_FOLDER
CPU_SRCS := $(wildcard $(CPU_FOLDER)/*.c)
# Create a matching list of binaries (replace .c with nothing, then prepend bin/CPU/)
CPU_BINS := $(patsubst $(CPU_FOLDER)/%.c, $(CPU_BIN_FOLDER)/%, $(CPU_SRCS))

# Default target builds all CPU programs
all: $(CPU_BINS)

# Compile each CPU binary with the time lib
$(CPU_BIN_FOLDER)/%: $(CPU_FOLDER)/%.c $(TIME_LIB_OBJ)
	@mkdir -p $(CPU_BIN_FOLDER)
	$(CC) $< -o $@ $(TIME_LIB_OBJ) -I$(INC_FOLDER) $(OPT) -lm

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

# Clean outputs
clean_batch_outputs:
	rm -f $(BATCH_OUT_FOLDER)/*

# Clean binaries and objects
clean:
	rm -rf $(BIN_FOLDER)
	rm -rf $(OBJ_FOLDER)