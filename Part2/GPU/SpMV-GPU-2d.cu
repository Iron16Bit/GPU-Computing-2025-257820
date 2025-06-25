#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

__global__
void spmv(int *Arows, int *Acols, double *Avals, double *v, double *C, int rows, int cols, int values) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < values) {
        double product = Avals[tid] * v[Acols[tid]];
        atomicAdd(&C[Arows[tid]], product);
    }
}

// Option 2: 2D tiled approach with advanced memory management
__global__
void spmv_2d_tiled_advanced(int *Arows, int *Acols, double *Avals, double *v, double *C,
                           int values, int tile_size) {
    // 2D grid of tiles
    int tile_x = blockIdx.x;
    int tile_y = blockIdx.y;
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    
    // Shared memory for tiling
    extern __shared__ double shared_data[];
    double *tile_vals = shared_data;
    double *tile_vector = &shared_data[blockDim.x * blockDim.y];
    int *tile_cols = (int*)&tile_vector[blockDim.x * blockDim.y];
    int *tile_rows = &tile_cols[blockDim.x * blockDim.y];
    
    // Calculate global indices with 2D mapping
    int elements_per_tile = blockDim.x * blockDim.y;
    int tile_start = (tile_y * gridDim.x + tile_x) * elements_per_tile;
    int local_idx = tid_y * blockDim.x + tid_x;
    int global_idx = tile_start + local_idx;
    
    // Cooperative loading into shared memory with banking optimization
    // Use different access patterns to avoid bank conflicts
    int bank_offset = (local_idx / 32) * 33; // Avoid bank conflicts
    
    if (global_idx < values) {
        tile_vals[local_idx] = Avals[global_idx];
        tile_cols[local_idx] = Acols[global_idx];
        tile_rows[local_idx] = Arows[global_idx];
    } else {
        tile_vals[local_idx] = 0.0;
        tile_cols[local_idx] = 0;
        tile_rows[local_idx] = -1;
    }
    __syncthreads();
    
    // Prefetch vector elements with 2D access pattern
    double result = 0.0;
    if (global_idx < values && tile_rows[local_idx] != -1) {
        // Use texture-like access pattern
        int col = tile_cols[local_idx];
        double v_val = v[col];
        result = tile_vals[local_idx] * v_val;
    }
    
    // 2D warp-level reduction
    // Reduce within warps first (32 threads)
    for (int offset = 16; offset > 0; offset /= 2) {
        double other_result = __shfl_down_sync(0xFFFFFFFF, result, offset);
        int other_row = __shfl_down_sync(0xFFFFFFFF, 
                                        (global_idx < values) ? tile_rows[local_idx] : -1, offset);
        if (other_row == tile_rows[local_idx] && tile_rows[local_idx] != -1) {
            result += other_result;
        }
    }
    
    // Write back with reduced atomic pressure
    if ((local_idx % 32) == 0 && global_idx < values && tile_rows[local_idx] != -1) {
        atomicAdd(&C[tile_rows[local_idx]], result);
    }
}

// Compute bandwidth and flops
void compute_band_gflops(int rows, int cols, int values, double time_ms, int* Acols) {
    // Bytes read from the COO
    size_t coo_size = (size_t)(sizeof(int) * (2 * values) + sizeof(double) * values);
    // Bytes read from the dense vector
    int* unique_cols = (int*)calloc(cols, sizeof(int));
    int unique_count = 0;
    for (int i=0; i<values; i++) {
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
    double bandwidth = total_bytes / (time_ms * 1.0e6);
    double operations = 2.0 * values;
    double gflops = operations / (time_ms * 1.0e6);

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
    
    // For 2D tiled SpMV
    // Define tile dimensions - e.g., 16x16 = 256 threads per block
    int tile_dim_x = 16; 
    int tile_dim_y = 16;
    int tile_size = tile_dim_x * tile_dim_y;

    // Calculate grid dimensions
    int tiles_x = (int)ceil(sqrt((double)values / tile_size));
    int tiles_y = (values + tiles_x * tile_size - 1) / (tiles_x * tile_size);

    dim3 threadsPerBlock(tile_dim_x, tile_dim_y);
    dim3 blocksPerGrid(tiles_x, tiles_y);

    size_t sharedMemSize = tile_size * (2 * sizeof(double) + 2 * sizeof(int));

    cudaEvent_t start, stop;

    first = 1;

    for (int i=0; i<ITERATIONS; i++) {
        cudaMemset(C, 0, rows * sizeof(double));
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);

        spmv_2d_tiled_advanced<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
        Arows, Acols, Avals, v, C, values, tile_size);
        
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