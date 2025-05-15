#include <stdlib.h>
#include <stdio.h>
#include <string.h>

__global__
void spmv(int *Arows, int *Acols, double *Avals, double *v, double *C, int rows, int cols, int values) {
    int current_row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (current_row < rows) {
        double sum = 0.0;
        
        // Binary search to find the starting position for this row
        int left = 0;
        int right = values - 1;
        int start_pos = values; // Default to end if row not found
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (Arows[mid] < current_row) {
                left = mid + 1;
            } else if (Arows[mid] > current_row) {
                right = mid - 1;
            } else {
                // Found a match, but we need to find the first occurrence
                start_pos = mid;
                right = mid - 1;
            }
        }
        
        // If no exact match found and left is valid, use it as start point
        if (start_pos == values || Arows[start_pos] != current_row) {
            start_pos = left;
        }
        
        // Accumulate products for this row
        for (int i = start_pos; i < values && Arows[i] == current_row; i++) {
            sum += Avals[i] * v[Acols[i]];
        }
        
        C[current_row] = sum;
    }
}

// Compute bandwidth and flops
void compute_band_gflops(int rows, int cols, int values, double time_ms, int* Acols) {
    // Bytes read from the COO
    size_t coo_size = (size_t)(sizeof(int) * (2 * values) + sizeof(double) * values);
    // Bytes read from the dense vector
    int* unique_cols = (int*)calloc(cols, sizeof(int));
    int unique_count = 0;
    for (int i=0; i<cols; i++) {
        if (unique_cols[Acols[i]] == 0) {
            unique_cols[Acols[i]] = 1;
            unique_count += 1;
        }
    }
    size_t vector_size = (size_t)(sizeof(double) * unique_count);
    // Total bytes read
    size_t bytes_read = coo_size + vector_size;
    // Bytes written
    size_t bytes_written = (size_t)(sizeof(double) * rows);
    size_t total_bytes = bytes_read + bytes_written;

    // GFLOPS
    double bandwidth = total_bytes / (time_ms * 1.0e9);
    double operations = 2.0 * values;
<<<<<<< HEAD
    double gflops = operations / (time_ms * 1.0e9);

=======
    
    // Convert to GFLOPS: operations / (time in seconds) / 1e9
    double gflops = operations / (time_ms / 1000.0) / 1e9;
    
    // Bandwidth calculation
    size_t bytes = sizeof(double) * (values + rows + cols) + sizeof(int) * (2 * values);
    // bytes * rows = each thread accesses that number of bytes, and # threads = # rows
    double bandwidth = ((bytes * rows) / 1e9) / (time_ms / 1000.0);
    
>>>>>>> 11679873ea01cdb6e7d99d7ae3b8c27b4c3b95ee
    printf("Bandwidth: %f GB/s\n", bandwidth);
    printf("FLOPS: %f GFLOPS\n", gflops);
}

void print_int_array(int* a, int n) {
    for (int i=0; i<n; i++) {
        printf("%d ", a[i]);
    }
    printf("\n");
}

void print_double_array(double* a, int n) {
    for (int i=0; i<n; i++) {
        printf("%f ", a[i]);
    }
    printf("\n");
}

void print_matrix(double* m, int rows, int cols) {
    for (int i=0; i<rows; i++) {
        for (int j=0; j<cols; j++) {
            printf("%f ", m[i*cols+j]);
        }
        printf("\n");
    }
}

#define ITERATIONS 51
#define DEFAULT_THREADS_PER_BLOCK 256

int main(int argc, char *argv[]) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "Usage: %s <input_file> [threads_per_block]\n", argv[0]);
        return 1;
    }

    // Parse threads per block parameter
    int threadsPerBlock = DEFAULT_THREADS_PER_BLOCK;
    if (argc == 3) {
        int user_threads = atoi(argv[2]);
        if (user_threads > 0) {
            threadsPerBlock = user_threads;
        } else {
            fprintf(stderr, "Warning: Invalid threads per block value, using default (%d)\n", 
                    DEFAULT_THREADS_PER_BLOCK);
        }
    }
    
    printf("Using %d threads per block\n", threadsPerBlock);
    printf("Used matrix: %s\n", argv[1]);

    FILE *fin = fopen(argv[1], "r");

    if (!fin) {
        perror("Failed to open file");
        return 1;
    }

    char buffer[100];
    int first = 1;
    double totalTime = 0.0;

    int rows;
    int cols;
    int values;

    int *Arows;
    int *Acols;
    double *Avals;

    int counter = 0;
    
    // Create COO from file
    while(fgets(buffer, 100, fin)) {
        if (buffer[0] != '%') {
            char *token = strtok(buffer, " ");
            char split_buffer[3][64];
            for (int i = 0; i < 3; i++) {
                if (!token) break;
                strncpy(split_buffer[i], token, 63);
                split_buffer[i][63] = '\0';
                token = strtok(NULL, " ");
            }
            if (first == 1) {
                first = 0;
                rows = atoi(split_buffer[0]);
                cols = atoi(split_buffer[1]);
                values = atoi(split_buffer[2]);
                
                // Use cudaMallocManaged instead of malloc
                cudaMallocManaged(&Arows, values*sizeof(int));
                cudaMallocManaged(&Acols, values*sizeof(int));
                cudaMallocManaged(&Avals, values*sizeof(double));
            } else {
                // Matrix Market files are 1-indexed
                int tmp_row = atoi(split_buffer[0])-1;
                int tmp_col = atoi(split_buffer[1])-1;
                double tmp_val = atof(split_buffer[2]);

                Arows[counter] = tmp_row;
                Acols[counter] = tmp_col;
                Avals[counter] = tmp_val;

                counter+=1;
            }
        }
    }

    // Create dense vector using cudaMallocManaged
    double *v;
    cudaMallocManaged(&v, cols*sizeof(double));
    for (int i=0; i<cols; i++) {
        v[i] = 1.0;
    }

    // Create output vector using cudaMallocManaged
    double *C;
    cudaMallocManaged(&C, rows*sizeof(double));
    
    // Perform SpMV
    int N = rows;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    first = 1;

    cudaEvent_t start, stop;

    for (int i=0; i<ITERATIONS; i++) {
        cudaMemset(C, 0, rows * sizeof(double));
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);

        spmv<<<blocksPerGrid, threadsPerBlock>>>(Arows, Acols, Avals, v, C, rows, cols, values);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float e_time = 0;
        cudaEventElapsedTime(&e_time, start, stop);
        // print_double_array(C, rows);
        // printf("Kernel completed in %fms\n", e_time);
        if (first == 1) {
            first = 0;
        } else {
            totalTime += e_time;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    // print_double_array(C, rows);

    // Calculate average time
    double avg_time = totalTime / (ITERATIONS - 1);
    printf("Average time: %fms\n", avg_time);
    compute_band_gflops(rows, cols, values, avg_time, Acols);

    fclose(fin);
    
    // Free using cudaFree instead of free
    cudaFree(Arows);
    cudaFree(Acols);
    cudaFree(Avals);
    cudaFree(v);
    cudaFree(C);

    return 0;
}