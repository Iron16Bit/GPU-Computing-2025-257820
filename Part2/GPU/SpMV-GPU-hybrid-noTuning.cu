#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <errno.h>
#include <math.h>
#include "my_time_lib.h"

// ============== CONFIGURABLE PARAMETERS ==============
#define DEFAULT_THREADS_PER_BLOCK 256
#define WARP_SIZE 32
#define ITERATIONS 51

// CUDA error checking macro
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

// ============== MATRIX STATISTICS AND ANALYSIS ==============

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
};

struct MAT_STATS calculate_matrix_stats(const struct CSR *matrix) {
    struct MAT_STATS stats = {0};
    
    stats.min_nnz_per_row = INT_MAX;
    stats.max_nnz_per_row = 0;
    
    double sum = 0.0;
    double sum_squares = 0.0;
    
    for (int i = 0; i < matrix->num_rows; i++) {
        int row_nnz = matrix->row_pointers[i + 1] - matrix->row_pointers[i];
        
        if (row_nnz == 0) {
            stats.empty_rows++;
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

// ============== ROW CLASSIFICATION ==============

void classify_rows(const int *row_ptr, int n, int **short_rows, int **long_rows, int **very_long_rows,
                   int *num_short, int *num_long, int *num_very_long, int threshold, int very_long_threshold) {
    // First pass: count short, long, and very long rows
    *num_short = 0;
    *num_long = 0;
    *num_very_long = 0;
    for (int i = 0; i < n; i++) {
        int row_length = row_ptr[i + 1] - row_ptr[i];
        if (row_length <= threshold) {
            (*num_short)++;
        } else if (row_length <= very_long_threshold) {
            (*num_long)++;
        } else {
            (*num_very_long)++;
        }
    }
    // Allocate arrays
    *short_rows = (int*)malloc(*num_short * sizeof(int));
    *long_rows = (int*)malloc(*num_long * sizeof(int));
    *very_long_rows = (int*)malloc(*num_very_long * sizeof(int));
    if (!*short_rows || !*long_rows || !*very_long_rows) {
        printf("Error: Failed to allocate memory for row classification\n");
        return;
    }
    // Second pass: populate arrays
    int short_idx = 0, long_idx = 0, very_long_idx = 0;
    for (int i = 0; i < n; i++) {
        int row_length = row_ptr[i + 1] - row_ptr[i];
        if (row_length <= threshold) {
            (*short_rows)[short_idx++] = i;
        } else if (row_length <= very_long_threshold) {
            (*long_rows)[long_idx++] = i;
        } else {
            (*very_long_rows)[very_long_idx++] = i;
        }
    }
}

// ============== ADAPTIVE LAUNCH CONFIGURATION ==============


void get_hybrid_launch_config(int n, int nnz, const int *row_ptr,
                             int &blocks, int &threads,
                             int **short_rows, int **long_rows, int **very_long_rows,
                             int *num_short, int *num_long, int *num_very_long,
                             int *short_blocks_limit, int *long_blocks_limit) {
    struct CSR temp_csr;
    temp_csr.row_pointers = (int*)row_ptr;
    temp_csr.num_rows = n;
    temp_csr.num_non_zeros = nnz;
    struct MAT_STATS stats = calculate_matrix_stats(&temp_csr);
    // printf("Matrix analysis:\n");
    // printf("  Mean NNZ per row: %.2f\n", stats.mean_nnz_per_row);
    // printf("  Std deviation: %.2f\n", stats.std_dev_nnz_per_row);
    // printf("  Max NNZ per row: %d\n", stats.max_nnz_per_row);
    // printf("  Empty rows: %d (%.2f%%)\n", stats.empty_rows, 100.0 * stats.empty_rows / n);
    int threshold, very_long_threshold;
    if (stats.mean_nnz_per_row < 3.0) {
        threads = 128;
        threshold = 16;
        very_long_threshold = 256; // For mawi, treat rows > 256 as 'very long'
        // printf("Strategy: Very sparse matrix - using 128 threads, threshold 16, very_long_threshold 256\n");
    } else if (stats.mean_nnz_per_row < 8.0) {
        threads = 128;
        threshold = 16;
        very_long_threshold = 256;
        // printf("Strategy: Sparse matrix - using 128 threads, threshold 16, very_long_threshold 256\n");
    } else if (stats.mean_nnz_per_row < 35.0) {
        if (stats.std_dev_nnz_per_row > 50.0) {
            threads = 256;
            threshold = 32;
            very_long_threshold = 512;
            // printf("Strategy: Medium density with high variance - using 256 threads, threshold 32, very_long_threshold 512\n");
        } else {
            threads = 256;
            threshold = 64;
            very_long_threshold = 512;
            // printf("Strategy: Medium density matrix - using 256 threads, threshold 64, very_long_threshold 512\n");
        }
    } else {
        threads = 128;
        threshold = 128;
        very_long_threshold = 1024;
        // printf("Strategy: Dense matrix - using 128 threads, threshold 128, very_long_threshold 1024\n");
    }
    classify_rows(row_ptr, n, short_rows, long_rows, very_long_rows, num_short, num_long, num_very_long, threshold, very_long_threshold);
    // printf("Final configuration: THREADS=%d THRESHOLD=%d VERY_LONG_THRESHOLD=%d\n", threads, threshold, very_long_threshold);
    // printf("Row classification (threshold=%d, very_long_threshold=%d):\n", threshold, very_long_threshold);
    // printf("  Short rows: %d (%.1f%%)\n", *num_short, 100.0 * (*num_short) / n);
    // printf("  Long rows: %d (%.1f%%)\n", *num_long, 100.0 * (*num_long) / n);
    // printf("  Very long rows: %d (%.1f%%)\n", *num_very_long, 100.0 * (*num_very_long) / n);
    *short_blocks_limit = (*num_short + threads - 1) / threads;
    *long_blocks_limit = (*num_long + (threads / WARP_SIZE) - 1) / (threads / WARP_SIZE);
    int very_long_blocks = *num_very_long; // 1 block per very long row
    blocks = *short_blocks_limit + *long_blocks_limit + very_long_blocks;
    // printf("Launch config: %d blocks (%d short + %d long + %d very long), %d threads\n",
    //        blocks, *short_blocks_limit, *long_blocks_limit, very_long_blocks, threads);
}

// ============== HYBRID KERNEL ==============

// Dual-strategy SpMV kernel with row-based partitioning

__global__ void hybrid_adaptive_spmv_optimized(const double *csr_values, const int *csr_row_ptr,
                                              const int *csr_col_indices, const double *vec,
                                              double *res, int n, const int *short_rows,
                                              const int *long_rows, const int *very_long_rows,
                                              int num_short, int num_long, int num_very_long,
                                              int short_blocks, int long_blocks, int block_offset = 0) {
    const int global_thread_id = (blockIdx.x + block_offset) * blockDim.x + threadIdx.x;
    const int warp_lane = global_thread_id & (WARP_SIZE - 1);
    const int effective_block_id = blockIdx.x + block_offset;
    
    // Strategy A: Thread-level processing for sparse rows
    if (effective_block_id < short_blocks) {
        const int sparse_row_index = global_thread_id;
        if (sparse_row_index >= num_short) return;
        const int target_row = short_rows[sparse_row_index];
        const int row_start_idx = csr_row_ptr[target_row];
        const int row_end_idx = csr_row_ptr[target_row + 1];
        const int elements_count = row_end_idx - row_start_idx;
        if (elements_count == 0) {
            res[target_row] = 0.0;
            return;
        }
        double accumulator = 0.0;
        int element_idx = row_start_idx;
        while (element_idx + 3 < row_end_idx) {
            accumulator += csr_values[element_idx]     * __ldg(&vec[csr_col_indices[element_idx]]);
            accumulator += csr_values[element_idx + 1] * __ldg(&vec[csr_col_indices[element_idx + 1]]);
            accumulator += csr_values[element_idx + 2] * __ldg(&vec[csr_col_indices[element_idx + 2]]);
            accumulator += csr_values[element_idx + 3] * __ldg(&vec[csr_col_indices[element_idx + 3]]);
            element_idx += 4;
        }
        while (element_idx < row_end_idx) {
            accumulator += csr_values[element_idx] * __ldg(&vec[csr_col_indices[element_idx]]);
            element_idx++;
        }
        res[target_row] = accumulator;
    }
    // Strategy B: Warp-cooperative processing for dense rows
    else if (effective_block_id < short_blocks + long_blocks) {
        const int warps_per_block = blockDim.x / WARP_SIZE;
        const int block_warp_id = threadIdx.x / WARP_SIZE;
        const int global_warp_id = (effective_block_id - short_blocks) * warps_per_block + block_warp_id;
        if (global_warp_id >= num_long) return;
        const int dense_row = long_rows[global_warp_id];
        const int row_begin = csr_row_ptr[dense_row];
        const int row_finish = csr_row_ptr[dense_row + 1];
        double partial_result = 0.0;
        for (int idx = row_begin + warp_lane; idx < row_finish; idx += WARP_SIZE) {
            partial_result += csr_values[idx] * __ldg(&vec[csr_col_indices[idx]]);
        }
        for (int stride = WARP_SIZE / 2; stride > 0; stride /= 2) {
            partial_result += __shfl_down_sync(0xFFFFFFFF, partial_result, stride);
        }
        if (warp_lane == 0) {
            res[dense_row] = partial_result;
        }
    }
    // Strategy C: Block-level processing for very long rows (optimized block-wide reduction)
    else {
        int very_long_row_idx = effective_block_id - short_blocks - long_blocks;
        if (very_long_row_idx >= num_very_long) return;
        int row = very_long_rows[very_long_row_idx];
        int row_start = csr_row_ptr[row];
        int row_end = csr_row_ptr[row + 1];
        double sum = 0.0;
        for (int idx = row_start + threadIdx.x; idx < row_end; idx += blockDim.x) {
            sum += csr_values[idx] * __ldg(&vec[csr_col_indices[idx]]);
        }
        // Block-wide reduction using shared memory
        extern __shared__ double sdata[];
        sdata[threadIdx.x] = sum;
        __syncthreads();
        // Reduce in shared memory
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                sdata[threadIdx.x] += sdata[threadIdx.x + s];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            res[row] = sdata[0];
        }
    }
}

// ============== MATRIX I/O AND CONVERSION ==============

void coo_to_csr(int *Arows, int *Acols, double *Avals, 
                int **row_ptr, int **csr_cols, double **csr_vals,
                int rows, int values) {
    
    CUDA_CHECK(cudaMallocManaged(row_ptr, (rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(csr_cols, values * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(csr_vals, values * sizeof(double)));
    
    // Initialize row_ptr
    memset(*row_ptr, 0, (rows + 1) * sizeof(int));
    
    // Count entries per row
    for (int i = 0; i < values; i++) {
        (*row_ptr)[Arows[i] + 1]++;
    }
    
    // Prefix sum
    for (int i = 1; i <= rows; i++) {
        (*row_ptr)[i] += (*row_ptr)[i - 1];
    }
    
    // Fill CSR arrays
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

// ============== UTILITY FUNCTIONS ==============

void print_result_vector(double* result, int n) {
    for (int i = 0; i < n; i++) {
        printf("%.6f\n", result[i]);
    }
}

// ============== PERFORMANCE METRICS ==============

void compute_performance_metrics(int rows, int cols, int values, double time_ms) {
    // Memory access calculation
    size_t matrix_bytes = sizeof(int) * (rows + 1) + sizeof(int) * values + sizeof(double) * values;
    size_t vector_bytes = sizeof(double) * cols;
    size_t result_bytes = sizeof(double) * rows;
    
    size_t total_bytes = matrix_bytes + vector_bytes + result_bytes;
    
    double bandwidth_gb_s = (total_bytes / (1024.0 * 1024.0 * 1024.0)) / (time_ms / 1000.0);
    double gflops = (2.0 * values) / (time_ms * 1.0e6);
    
    printf("Execution time: %.3f ms\n", time_ms);
    printf("Bandwidth: %.2f GB/s", bandwidth_gb_s);
    
    if (bandwidth_gb_s > 800.0) {
        printf(" (WARNING: Bandwidth exceeds realistic limits)");
    }
    printf("\n");
    
    printf("GFLOPS: %.2f\n", gflops);
}

// ============== MAIN FUNCTION ==============

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
        return 1;
    }
    printf("Matrix: %s\n\n", argv[1]);
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
    int blocks, threads;
    int *short_rows, *long_rows, *very_long_rows;
    int num_short, num_long, num_very_long, short_blocks_limit, long_blocks_limit;
    get_hybrid_launch_config(rows, values, row_ptr, blocks, threads,
                            &short_rows, &long_rows, &very_long_rows,
                            &num_short, &num_long, &num_very_long,
                            &short_blocks_limit, &long_blocks_limit);
    TIMER_STOP(var);
    time = TIMER_ELAPSED(var) / 1000.0;
    printf("Preprocessing time: %.3f ms\n", time);
    int *d_short_rows = nullptr, *d_long_rows = nullptr, *d_very_long_rows = nullptr;
    if (num_short > 0) {
        CUDA_CHECK(cudaMalloc(&d_short_rows, num_short * sizeof(int)));
        checkCudaError("cudaMalloc d_short_rows");
        CUDA_CHECK(cudaMemcpy(d_short_rows, short_rows, num_short * sizeof(int), cudaMemcpyHostToDevice));
        checkCudaError("cudaMemcpy d_short_rows");
    }
    if (num_long > 0) {
        CUDA_CHECK(cudaMalloc(&d_long_rows, num_long * sizeof(int)));
        checkCudaError("cudaMalloc d_long_rows");
        CUDA_CHECK(cudaMemcpy(d_long_rows, long_rows, num_long * sizeof(int), cudaMemcpyHostToDevice));
        checkCudaError("cudaMemcpy d_long_rows");
    }
    if (num_very_long > 0) {
        CUDA_CHECK(cudaMalloc(&d_very_long_rows, num_very_long * sizeof(int)));
        checkCudaError("cudaMalloc d_very_long_rows");
        CUDA_CHECK(cudaMemcpy(d_very_long_rows, very_long_rows, num_very_long * sizeof(int), cudaMemcpyHostToDevice));
        checkCudaError("cudaMemcpy d_very_long_rows");
    }
    cudaEvent_t start, stop;
    double totalTime = 0.0;
    // Parameters for very long rows
    int very_long_block_size = 1024;
    size_t very_long_shared_mem = very_long_block_size * sizeof(double);
    int very_long_blocks = num_very_long;
    int regular_blocks = short_blocks_limit + long_blocks_limit;
    for (int i = 0; i < ITERATIONS; i++) {
        CUDA_CHECK(cudaMemset(C, 0, rows * sizeof(double)));
        checkCudaError("cudaMemset C");
        CUDA_CHECK(cudaEventCreate(&start));
        checkCudaError("cudaEventCreate start");
        CUDA_CHECK(cudaEventCreate(&stop));
        checkCudaError("cudaEventCreate stop");
        CUDA_CHECK(cudaEventRecord(start));
        checkCudaError("cudaEventRecord start");
        
        // Launch short and long rows together (they use the same thread configuration)
        if (regular_blocks > 0) {
            hybrid_adaptive_spmv_optimized<<<regular_blocks, threads>>>(
                csr_vals, row_ptr, csr_cols, v, C, rows,
                d_short_rows, d_long_rows, d_very_long_rows,
                num_short, num_long, num_very_long,
                short_blocks_limit, long_blocks_limit, 0);
            checkCudaError("hybrid_adaptive_spmv_optimized (short/long)");
        }
        
        // Launch very long rows separately with optimal block size and correct offset
        if (very_long_blocks > 0) {
            hybrid_adaptive_spmv_optimized<<<very_long_blocks, very_long_block_size, very_long_shared_mem>>>(
                csr_vals, row_ptr, csr_cols, v, C, rows,
                d_short_rows, d_long_rows, d_very_long_rows,
                num_short, num_long, num_very_long,
                short_blocks_limit, long_blocks_limit, regular_blocks);
            checkCudaError("hybrid_adaptive_spmv_optimized (very long)");
        }
        CUDA_CHECK(cudaEventRecord(stop));
        checkCudaError("cudaEventRecord stop");
        CUDA_CHECK(cudaEventSynchronize(stop));
        checkCudaError("cudaEventSynchronize stop");
        float elapsed_time;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        checkCudaError("cudaEventElapsedTime");
        if (i > 0) {
            totalTime += elapsed_time;
        }
        CUDA_CHECK(cudaEventDestroy(start));
        checkCudaError("cudaEventDestroy start");
        CUDA_CHECK(cudaEventDestroy(stop));
        checkCudaError("cudaEventDestroy stop");
    }
    double avg_time = totalTime / (ITERATIONS - 1);
    compute_performance_metrics(rows, cols, values, avg_time);
    
    // Quick verification - check a few result values to ensure computation was correct
    // printf("\nQuick verification (first 5 non-zero results):\n");
    // int printed = 0;
    // for (int i = 0; i < rows && printed < 5; i++) {
    //     if (C[i] != 0.0) {
    //         printf("  C[%d] = %.6f\n", i, C[i]);
    //         printed++;
    //     }
    // }
    // if (printed == 0) {
    //     printf("  WARNING: All results are zero - possible computation error!\n");
    // }
    
    // printf("\nResult vector (one value per line):\n");
    // print_result_vector(C, rows);
    CUDA_CHECK(cudaFree(Arows));
    CUDA_CHECK(cudaFree(Acols));
    CUDA_CHECK(cudaFree(Avals));
    CUDA_CHECK(cudaFree(row_ptr));
    CUDA_CHECK(cudaFree(csr_cols));
    CUDA_CHECK(cudaFree(csr_vals));
    CUDA_CHECK(cudaFree(v));
    CUDA_CHECK(cudaFree(C));
    return 0;
}