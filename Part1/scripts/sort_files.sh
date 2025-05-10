#!/bin/bash

# Path to your sort_mtx executable
SORT_BIN=./scripts/sort

# Folder containing .mtx files
DATA_DIR=Data

# Check if executable exists
if [ ! -x "$SORT_BIN" ]; then
    gcc -O3 -march=native -o ./scripts/sort ./scripts/sort.c
fi

# Process each .mtx file
for file in "$DATA_DIR"/*.mtx; do
    echo "Sorting $file..."
    "$SORT_BIN" "$file"
done
