#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <errno.h>
#include <math.h>
#include <cuda.h>
#include "my_time_lib.h"

#define DEFAULT_THREADS_PER_BLOCK 1024
#define WARP_SIZE 32
#define ITERATIONS 51
#define MAX_ROW_SPLIT_SIZE 4096
#define very_long_THRESHOLD 8192
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

void checkCudaError(const char* msg) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error after %s: %s\n", msg, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

// ============== STRUCTURES ==============

struct CSR {
    int *row_pointers;
    int *col_indices;
    double *values;
    int num_rows;
    int num_cols;
    int num_non_zeros;
};

struct MAT_STATS {
    double mean_nnz_per_row;
    double std_dev_nnz_per_row;
    int max_nnz_per_row;
    int min_nnz_per_row;
    int empty_rows;
    double variance_nnz_per_row;
    int very_long_rows;
};

struct ROW_CHUNK {
    int row_id;
    int chunk_start;
    int chunk_end;
    int chunk_id;
    int total_chunks;
};

struct MAT_STATS calculate_enhanced_matrix_stats(const struct CSR *matrix) {
    struct MAT_STATS stats = {0};
    
    stats.min_nnz_per_row = INT_MAX;
    stats.max_nnz_per_row = 0;
    stats.very_long_rows = 0;
    
    double sum = 0.0;
    double sum_squares = 0.0;
    
    for (int i = 0; i < matrix->num_rows; i++) {
        int row_nnz = matrix->row_pointers[i + 1] - matrix->row_pointers[i];
        
        if (row_nnz == 0) {
            stats.empty_rows++;
        }
        
        if (row_nnz > very_long_THRESHOLD) {
            stats.very_long_rows++;
        }
        
        if (row_nnz < stats.min_nnz_per_row) {
            stats.min_nnz_per_row = row_nnz;
        }
        if (row_nnz > stats.max_nnz_per_row) {
            stats.max_nnz_per_row = row_nnz;
        }
        
        sum += row_nnz;
        sum_squares += row_nnz * row_nnz;
    }
    
    stats.mean_nnz_per_row = sum / matrix->num_rows;
    stats.variance_nnz_per_row = (sum_squares / matrix->num_rows) - (stats.mean_nnz_per_row * stats.mean_nnz_per_row);
    stats.std_dev_nnz_per_row = sqrt(stats.variance_nnz_per_row);
    
    return stats;
}

void classify_and_split_rows(const int *row_ptr, int n, 
                             int **short_rows, int **long_rows, 
                             ROW_CHUNK **very_long_chunks,
                             int *num_short, int *num_long, int *num_very_long_chunks,
                             int short_threshold, int long_threshold) {
    
    *num_short = 0;
    *num_long = 0;
    *num_very_long_chunks = 0;
    
    std::vector<ROW_CHUNK> temp_chunks;
    int very_long_rows_count = 0;
    int max_chunks_for_single_row = 0;
    int total_very_long_elements = 0;
    
    for (int i = 0; i < n; i++) {
        int row_length = row_ptr[i + 1] - row_ptr[i];
        
        if (row_length <= short_threshold) {
            (*num_short)++;
        } else if (row_length <= long_threshold) {
            (*num_long)++;
        } else {
            very_long_rows_count++;
            total_very_long_elements += row_length;
            
            int chunks_needed = (row_length + MAX_ROW_SPLIT_SIZE - 1) / MAX_ROW_SPLIT_SIZE;
            max_chunks_for_single_row = max(max_chunks_for_single_row, chunks_needed);
            
            for (int chunk = 0; chunk < chunks_needed; chunk++) {
                ROW_CHUNK rc;
                rc.row_id = i;
                rc.chunk_start = row_ptr[i] + chunk * MAX_ROW_SPLIT_SIZE;
                rc.chunk_end = min(rc.chunk_start + MAX_ROW_SPLIT_SIZE, row_ptr[i + 1]);
                rc.chunk_id = chunk;
                rc.total_chunks = chunks_needed;
                temp_chunks.push_back(rc);
            }
            *num_very_long_chunks += chunks_needed;
        }
    }
    
    *short_rows = (int*)malloc(*num_short * sizeof(int));
    *long_rows = (int*)malloc(*num_long * sizeof(int));
    *very_long_chunks = (ROW_CHUNK*)malloc(*num_very_long_chunks * sizeof(ROW_CHUNK));
    
    if (!*short_rows || !*long_rows || !*very_long_chunks) {
        printf("Error: Failed to allocate memory for enhanced row classification\n");
        return;
    }
    
    int short_idx = 0, long_idx = 0;
    
    for (int i = 0; i < n; i++) {
        int row_length = row_ptr[i + 1] - row_ptr[i];
        
        if (row_length <= short_threshold) {
            (*short_rows)[short_idx++] = i;
        } else if (row_length <= long_threshold) {
            (*long_rows)[long_idx++] = i;
        }
    }
    
    for (size_t i = 0; i < temp_chunks.size(); i++) {
        (*very_long_chunks)[i] = temp_chunks[i];
    }
}

struct CONFIG_OPTION {
    const char* name;
    int short_threshold;
    int long_threshold;
    int threads_per_block;
    const char* description;
};

void get_enhanced_launch_config(int n, int nnz, const int *row_ptr,
                               int &total_blocks, int &threads,
                               int **short_rows, int **long_rows, 
                               ROW_CHUNK **very_long_chunks,
                               int *num_short, int *num_long, int *num_very_long_chunks,
                               int *short_blocks, int *long_blocks, int *very_long_blocks) {
    
    struct CSR temp_csr;
    temp_csr.row_pointers = (int*)row_ptr;
    temp_csr.num_rows = n;
    temp_csr.num_non_zeros = nnz;
    struct MAT_STATS stats = calculate_enhanced_matrix_stats(&temp_csr);
    
    printf("Matrix Statistics:\n");
    printf("  Rows: %d, NNZ: %d\n", n, nnz);
    printf("  Mean NNZ/row: %.2f, Std Dev: %.2f\n", 
           stats.mean_nnz_per_row, stats.std_dev_nnz_per_row);
    printf("  Max NNZ/row: %d, Min NNZ/row: %d\n", stats.max_nnz_per_row, stats.min_nnz_per_row);
    printf("  Empty rows: %d (%.1f%%)\n", stats.empty_rows, 100.0 * stats.empty_rows / n);
    printf("\n");
    
    // {name, short_threshold, long_threshold, threads_per_block, description}
    CONFIG_OPTION configs[] = {
        {"Short-Dense", 16, 64, 1024, "Optimized for short dense rows"}, // More rows as short for uniform small matrices
        {"Balanced", 32, 128, 1024, "Balanced for medium row lengths"}, // Good balance for most matrices
        {"Dense", 64, 256, 1024, "Optimized for very dense matrices"} // High thresholds to avoid atomic overhead
    };
    
    int selected_config = 1; // Default to Balanced
    
    printf("Auto-Selection Logic:\n");
    
    // Improved selection logic based on matrix characteristics
    if (stats.mean_nnz_per_row < 15.0 || (stats.mean_nnz_per_row < 35.0 && stats.max_nnz_per_row < 100)) {
        selected_config = 0; // Short-Dense
        printf("  Matrix has short rows (mean=%.2f) -> Selecting Short-Dense\n", stats.mean_nnz_per_row);
    } else if (stats.mean_nnz_per_row > 100.0) {
        selected_config = 3; // Dense
        printf("  Matrix is very dense (mean=%.2f) -> Selecting Dense\n", stats.mean_nnz_per_row);
    } else {
        selected_config = 1; // Balanced
        printf("  Matrix has medium density (mean=%.2f, std=%.2f) -> Selecting Balanced\n", 
                stats.mean_nnz_per_row, stats.std_dev_nnz_per_row);
    }
    
    CONFIG_OPTION chosen = configs[selected_config];
    threads = chosen.threads_per_block;
    
    printf("\nSelected Configuration: %s\n", chosen.name);
    printf("  Short threshold: <= %d elements\n", chosen.short_threshold);
    printf("  Long threshold: <= %d elements\n", chosen.long_threshold);
    printf("  Threads per block: %d\n", chosen.threads_per_block);
    printf("  Strategy: %s\n", chosen.description);
    
    classify_and_split_rows(row_ptr, n, short_rows, long_rows, very_long_chunks,
                           num_short, num_long, num_very_long_chunks,
                           chosen.short_threshold, chosen.long_threshold);
    
    printf("\nFinal Classification:\n");
    printf("  Short rows: %d (%.1f%%) - 1 thread per row\n",
           *num_short, 100.0 * (*num_short) / n);
    printf("  Long rows: %d (%.1f%%) - Warp-cooperative processing\n",
           *num_long, 100.0 * (*num_long) / n);
    printf("  Ultra long chunks: %d - Block-level parallel processing\n", *num_very_long_chunks);
    
    *short_blocks = (*num_short + threads - 1) / threads;
    *long_blocks = (*num_long + (threads / WARP_SIZE) - 1) / (threads / WARP_SIZE);
    *very_long_blocks = *num_very_long_chunks;
    
    total_blocks = *short_blocks + *long_blocks + *very_long_blocks;
    
    printf("\nLaunch Configuration:\n");
    printf("  Short rows: %d blocks x %d threads = %d total threads\n", 
           *short_blocks, threads, *short_blocks * threads);
    printf("  Long rows: %d blocks x %d threads (%d warps)\n", 
           *long_blocks, threads, *long_blocks * (threads / WARP_SIZE));
    printf("  Ultra long chunks: %d blocks x 256 threads\n", *very_long_blocks);
    printf("  Total blocks: %d\n", total_blocks);
    printf("\n");
}

__global__ void spmv(const double *__restrict__ csr_values, 
                                     const int *__restrict__ csr_row_ptr,
                                     const int *__restrict__ csr_col_indices, 
                                     const double *__restrict__ vec,
                                     double *__restrict__ res, int n, 
                                     const int *__restrict__ short_rows,
                                     const int *__restrict__ long_rows, 
                                     const ROW_CHUNK *__restrict__ very_long_chunks,
                                     int num_short, int num_long, int num_very_long_chunks,
                                     int short_blocks, int long_blocks, int very_long_blocks) {
    
    const int block_id = blockIdx.x;
    const int thread_id = threadIdx.x;
    const int warp_lane = thread_id & (WARP_SIZE - 1);
    const int warp_id = thread_id >> 5;
    
    if (block_id < short_blocks) {
        const int global_thread_id = block_id * blockDim.x + thread_id;
        if (global_thread_id >= num_short) return;
        
        const int row = short_rows[global_thread_id];
        const int start = csr_row_ptr[row];
        const int end = csr_row_ptr[row + 1];
        
        double sum = 0.0;
        
        #pragma unroll 4
        for (int idx = start; idx < end; idx++) {
            sum += csr_values[idx] * __ldg(&vec[csr_col_indices[idx]]);
        }
        res[row] = sum;
    }
    else if (block_id < short_blocks + long_blocks) {
        int warp_global_id = (block_id - short_blocks) * (blockDim.x >> 5) + warp_id;
        
        if (warp_global_id < num_long) {
            int row = long_rows[warp_global_id];
            int start = csr_row_ptr[row];
            int end = csr_row_ptr[row + 1];
            
            double thread_sum = 0.0;
            
            for (int idx = start + warp_lane; idx < end; idx += WARP_SIZE) {
                thread_sum += csr_values[idx] * __ldg(&vec[csr_col_indices[idx]]);
            }
            
            #pragma unroll
            for (int stride = 16; stride > 0; stride >>= 1) {
                thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, stride);
            }
            
            if (warp_lane == 0) {
                res[row] = thread_sum;
            }
        }
    }
    else if (block_id < short_blocks + long_blocks + very_long_blocks) {
        const int very_long_idx = block_id - short_blocks - long_blocks;
        if (very_long_idx >= num_very_long_chunks) return;
        
        const ROW_CHUNK chunk = very_long_chunks[very_long_idx];
        const int row = chunk.row_id;
        const int start = chunk.chunk_start;
        const int end = chunk.chunk_end;
        const int row_length = end - start;
        
        const int lane_id = thread_id & (WARP_SIZE - 1);
        
        double thread_sum = 0.0;
        
        const int elements_per_thread = (row_length > 2048) ? 8 : 4;
        const int chunk_size = blockDim.x * elements_per_thread;
        const int total_chunks = (row_length + chunk_size - 1) / chunk_size;
        
        for (int chunk = 0; chunk < total_chunks; chunk++) {
            int chunk_start = start + chunk * chunk_size;
            int thread_start = chunk_start + thread_id * elements_per_thread;
            
            if (elements_per_thread == 8) {
                if (thread_start < end) 
                    thread_sum += csr_values[thread_start] * __ldg(&vec[csr_col_indices[thread_start]]);
                if (thread_start + 1 < end) 
                    thread_sum += csr_values[thread_start + 1] * __ldg(&vec[csr_col_indices[thread_start + 1]]);
                if (thread_start + 2 < end) 
                    thread_sum += csr_values[thread_start + 2] * __ldg(&vec[csr_col_indices[thread_start + 2]]);
                if (thread_start + 3 < end) 
                    thread_sum += csr_values[thread_start + 3] * __ldg(&vec[csr_col_indices[thread_start + 3]]);
                if (thread_start + 4 < end) 
                    thread_sum += csr_values[thread_start + 4] * __ldg(&vec[csr_col_indices[thread_start + 4]]);
                if (thread_start + 5 < end) 
                    thread_sum += csr_values[thread_start + 5] * __ldg(&vec[csr_col_indices[thread_start + 5]]);
                if (thread_start + 6 < end) 
                    thread_sum += csr_values[thread_start + 6] * __ldg(&vec[csr_col_indices[thread_start + 6]]);
                if (thread_start + 7 < end) 
                    thread_sum += csr_values[thread_start + 7] * __ldg(&vec[csr_col_indices[thread_start + 7]]);
            } else {
                if (thread_start < end) 
                    thread_sum += csr_values[thread_start] * __ldg(&vec[csr_col_indices[thread_start]]);
                if (thread_start + 1 < end) 
                    thread_sum += csr_values[thread_start + 1] * __ldg(&vec[csr_col_indices[thread_start + 1]]);
                if (thread_start + 2 < end) 
                    thread_sum += csr_values[thread_start + 2] * __ldg(&vec[csr_col_indices[thread_start + 2]]);
                if (thread_start + 3 < end) 
                    thread_sum += csr_values[thread_start + 3] * __ldg(&vec[csr_col_indices[thread_start + 3]]);
            }
        }
        
        #pragma unroll
        for (int stride = 16; stride > 0; stride >>= 1) {
            thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, stride);
        }
        
        if (lane_id == 0) {
            atomicAdd(&res[row], thread_sum);
        }
    }
}

void coo_to_csr(int *Arows, int *Acols, double *Avals, 
                int **row_ptr, int **csr_cols, double **csr_vals,
                int rows, int values) {
    
    CUDA_CHECK(cudaMallocManaged(row_ptr, (rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(csr_cols, values * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(csr_vals, values * sizeof(double)));
    
    memset(*row_ptr, 0, (rows + 1) * sizeof(int));
    
    for (int i = 0; i < values; i++) {
        (*row_ptr)[Arows[i] + 1]++;
    }
    
    for (int i = 1; i <= rows; i++) {
        (*row_ptr)[i] += (*row_ptr)[i - 1];
    }
    
    std::vector<int> temp_ptr(rows + 1);
    for (int i = 0; i <= rows; i++) {
        temp_ptr[i] = (*row_ptr)[i];
    }
    
    for (int i = 0; i < values; i++) {
        int row = Arows[i];
        int pos = temp_ptr[row]++;
        (*csr_cols)[pos] = Acols[i];
        (*csr_vals)[pos] = Avals[i];
    }
}

void compute_performance_metrics(int rows, int cols, int values, double time_ms) {
    size_t matrix_bytes = sizeof(int) * (rows + 1) + sizeof(int) * values + sizeof(double) * values;
    size_t vector_bytes = sizeof(double) * cols;
    size_t result_bytes = sizeof(double) * rows;
    
    size_t total_bytes = matrix_bytes + vector_bytes + result_bytes;
    
    double bandwidth_gb_s = (total_bytes / (1024.0 * 1024.0 * 1024.0)) / (time_ms / 1000.0);
    double gflops = (2.0 * values) / (time_ms * 1.0e6);
    
    printf("Performance Metrics:\n");
    printf("  Execution time: %.3f ms\n", time_ms);
    printf("  Bandwidth: %.2f GB/s\n", bandwidth_gb_s);
    printf("  GFLOPS: %.2f\n", gflops);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
        return 1;
    }
    
    FILE *fin = fopen(argv[1], "r");
    if (!fin) {
        fprintf(stderr, "Error: Failed to open file '%s': %s\n", argv[1], strerror(errno));
        return 1;
    }
    
    char buffer[256];
    int first = 1;
    int rows, cols, values;
    int *Arows, *Acols;
    double *Avals;
    int counter = 0;
    
    while(fgets(buffer, sizeof(buffer), fin)) {
        if (buffer[0] != '%') {
            if (first == 1) {
                if (sscanf(buffer, "%d %d %d", &rows, &cols, &values) != 3) {
                    fprintf(stderr, "Error: Failed to parse matrix dimensions\n");
                    fclose(fin);
                    return 1;
                }
                if (rows <= 0 || cols <= 0 || values <= 0) {
                    fprintf(stderr, "Error: Invalid matrix dimensions\n");
                    fclose(fin);
                    return 1;
                }
                first = 0;
                CUDA_CHECK(cudaMallocManaged(&Arows, values * sizeof(int)));
                CUDA_CHECK(cudaMallocManaged(&Acols, values * sizeof(int)));
                CUDA_CHECK(cudaMallocManaged(&Avals, values * sizeof(double)));
            } else {
                int row, col;
                double val;
                if (sscanf(buffer, "%d %d %lf", &row, &col, &val) != 3) {
                    fprintf(stderr, "Error: Failed to parse matrix entry\n");
                    fclose(fin);
                    return 1;
                }
                Arows[counter] = row - 1;
                Acols[counter] = col - 1;
                Avals[counter] = val;
                counter++;
            }
        }
    }
    fclose(fin);
    
    if (counter != values) {
        fprintf(stderr, "Error: Expected %d values but read %d\n", values, counter);
        return 1;
    }
    
    int *row_ptr, *csr_cols;
    double *csr_vals;
    coo_to_csr(Arows, Acols, Avals, &row_ptr, &csr_cols, &csr_vals, rows, values);
    
    double *v, *C;
    CUDA_CHECK(cudaMallocManaged(&v, cols * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&C, rows * sizeof(double)));
    
    for (int i = 0; i < cols; i++) {
        v[i] = 1.0;
    }
    
    double time;
    TIMER_DEF(var);
    TIMER_START(var);
    
    int total_blocks, threads;
    int *short_rows, *long_rows;
    ROW_CHUNK *very_long_chunks;
    int num_short, num_long, num_very_long_chunks;
    int short_blocks, long_blocks, very_long_blocks;
    
    get_enhanced_launch_config(rows, values, row_ptr, total_blocks, threads,
                              &short_rows, &long_rows, &very_long_chunks,
                              &num_short, &num_long, &num_very_long_chunks,
                              &short_blocks, &long_blocks, &very_long_blocks);
    
    TIMER_STOP(var);
    time = TIMER_ELAPSED(var) / 1000.0;
    printf("Preprocessing time: %.3f ms\n", time);
    
    int *d_short_rows = nullptr, *d_long_rows = nullptr;
    ROW_CHUNK *d_very_long_chunks = nullptr;
    
    if (num_short > 0) {
        CUDA_CHECK(cudaMalloc(&d_short_rows, num_short * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_short_rows, short_rows, num_short * sizeof(int), cudaMemcpyHostToDevice));
    }
    if (num_long > 0) {
        CUDA_CHECK(cudaMalloc(&d_long_rows, num_long * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_long_rows, long_rows, num_long * sizeof(int), cudaMemcpyHostToDevice));
    }
    if (num_very_long_chunks > 0) {
        CUDA_CHECK(cudaMalloc(&d_very_long_chunks, num_very_long_chunks * sizeof(ROW_CHUNK)));
        CUDA_CHECK(cudaMemcpy(d_very_long_chunks, very_long_chunks, num_very_long_chunks * sizeof(ROW_CHUNK), cudaMemcpyHostToDevice));
    }
    
    cudaEvent_t start, stop;
    double totalTime = 0.0;
    
    int total_kernel_blocks = short_blocks + long_blocks + very_long_blocks;
    
    int unified_threads = 1024;
    
    for (int i = 0; i < ITERATIONS; i++) {
        CUDA_CHECK(cudaMemset(C, 0, rows * sizeof(double)));
        
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
        
        if (total_kernel_blocks > 0) {
            spmv<<<total_kernel_blocks, unified_threads>>>(
                csr_vals, row_ptr, csr_cols, v, C, rows,
                d_short_rows, d_long_rows, d_very_long_chunks,
                num_short, num_long, num_very_long_chunks,
                short_blocks, long_blocks, very_long_blocks);
            checkCudaError("spmv (all strategies)");
        }
        
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float elapsed_time;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        
        if (i > 0) {
            totalTime += elapsed_time;
        }
        
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    
    double avg_time = totalTime / (ITERATIONS - 1);
    compute_performance_metrics(rows, cols, values, avg_time);
    
    CUDA_CHECK(cudaFree(Arows));
    CUDA_CHECK(cudaFree(Acols));
    CUDA_CHECK(cudaFree(Avals));
    CUDA_CHECK(cudaFree(row_ptr));
    CUDA_CHECK(cudaFree(csr_cols));
    CUDA_CHECK(cudaFree(csr_vals));
    CUDA_CHECK(cudaFree(v));
    CUDA_CHECK(cudaFree(C));
    
    if (d_short_rows) CUDA_CHECK(cudaFree(d_short_rows));
    if (d_long_rows) CUDA_CHECK(cudaFree(d_long_rows));
    if (d_very_long_chunks) CUDA_CHECK(cudaFree(d_very_long_chunks));
    
    free(short_rows);
    free(long_rows);
    free(very_long_chunks);
    
    return 0;
}
