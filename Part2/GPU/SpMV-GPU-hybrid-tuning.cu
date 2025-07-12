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
#define MAX_SHARED_MEM_PER_BLOCK 48000  // 48KB shared memory
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

// GPU kernel for calculating row lengths and statistics
__global__ void calculate_row_stats_kernel(const int *row_ptr, int num_rows, 
                                           int *row_lengths, int *min_val, int *max_val, 
                                           int *empty_count, double *sum, double *sum_squares) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Shared memory for block-level reductions
    __shared__ int s_min[256];
    __shared__ int s_max[256];
    __shared__ int s_empty[256];
    __shared__ double s_sum[256];
    __shared__ double s_sum_sq[256];
    
    int local_min = INT_MAX;
    int local_max = 0;
    int local_empty = 0;
    double local_sum = 0.0;
    double local_sum_sq = 0.0;
    
    if (idx < num_rows) {
        int row_nnz = row_ptr[idx + 1] - row_ptr[idx];
        row_lengths[idx] = row_nnz;
        
        local_min = row_nnz;
        local_max = row_nnz;
        local_empty = (row_nnz == 0) ? 1 : 0;
        local_sum = (double)row_nnz;
        local_sum_sq = (double)(row_nnz * row_nnz);
    }
    
    int tid = threadIdx.x;
    s_min[tid] = local_min;
    s_max[tid] = local_max;
    s_empty[tid] = local_empty;
    s_sum[tid] = local_sum;
    s_sum_sq[tid] = local_sum_sq;
    __syncthreads();
    
    // Block-level reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_min[tid] = min(s_min[tid], s_min[tid + stride]);
            s_max[tid] = max(s_max[tid], s_max[tid + stride]);
            s_empty[tid] += s_empty[tid + stride];
            s_sum[tid] += s_sum[tid + stride];
            s_sum_sq[tid] += s_sum_sq[tid + stride];
        }
        __syncthreads();
    }
    
    // Store block results
    if (tid == 0) {
        atomicMin(min_val, s_min[0]);
        atomicMax(max_val, s_max[0]);
        atomicAdd(empty_count, s_empty[0]);
        atomicAdd(sum, s_sum[0]);
        atomicAdd(sum_squares, s_sum_sq[0]);
    }
}

struct MAT_STATS calculate_matrix_stats_gpu(const int *row_ptr, int num_rows) {
    struct MAT_STATS stats = {0};
    
    // Allocate GPU memory for statistics
    int *d_row_lengths, *d_min_val, *d_max_val, *d_empty_count;
    double *d_sum, *d_sum_squares;
    
    CUDA_CHECK(cudaMalloc(&d_row_lengths, num_rows * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_min_val, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_max_val, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_empty_count, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sum_squares, sizeof(double)));
    
    // Initialize values
    int init_min = INT_MAX, init_max = 0, init_empty = 0;
    double init_sum = 0.0, init_sum_sq = 0.0;
    CUDA_CHECK(cudaMemcpy(d_min_val, &init_min, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_max_val, &init_max, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_empty_count, &init_empty, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sum, &init_sum, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sum_squares, &init_sum_sq, sizeof(double), cudaMemcpyHostToDevice));
    
    // Launch kernel
    int threads = 256;
    int blocks = (num_rows + threads - 1) / threads;
    calculate_row_stats_kernel<<<blocks, threads>>>(row_ptr, num_rows, d_row_lengths, 
                                                   d_min_val, d_max_val, d_empty_count, 
                                                   d_sum, d_sum_squares);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy results back
    int min_val, max_val, empty_count;
    double sum, sum_squares;
    CUDA_CHECK(cudaMemcpy(&min_val, d_min_val, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&max_val, d_max_val, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&empty_count, d_empty_count, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sum_squares, d_sum_squares, sizeof(double), cudaMemcpyDeviceToHost));
    
    // Calculate final statistics
    stats.min_nnz_per_row = min_val;
    stats.max_nnz_per_row = max_val;
    stats.empty_rows = empty_count;
    stats.mean_nnz_per_row = sum / num_rows;
    stats.variance_nnz_per_row = (sum_squares / num_rows) - (stats.mean_nnz_per_row * stats.mean_nnz_per_row);
    stats.std_dev_nnz_per_row = sqrt(stats.variance_nnz_per_row);
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_row_lengths));
    CUDA_CHECK(cudaFree(d_min_val));
    CUDA_CHECK(cudaFree(d_max_val));
    CUDA_CHECK(cudaFree(d_empty_count));
    CUDA_CHECK(cudaFree(d_sum));
    CUDA_CHECK(cudaFree(d_sum_squares));
    
    return stats;
}

// ============== ROW CLASSIFICATION ==============

// GPU kernel for row classification
__global__ void classify_rows_kernel(const int *row_ptr, int n, int threshold, int very_long_threshold,
                                   int *short_rows, int *long_rows, int *very_long_rows,
                                   int *short_count, int *long_count, int *very_long_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        int row_length = row_ptr[idx + 1] - row_ptr[idx];
        
        if (row_length <= threshold) {
            int pos = atomicAdd(short_count, 1);
            short_rows[pos] = idx;
        } else if (row_length <= very_long_threshold) {
            int pos = atomicAdd(long_count, 1);
            long_rows[pos] = idx;
        } else {
            int pos = atomicAdd(very_long_count, 1);
            very_long_rows[pos] = idx;
        }
    }
}

// GPU kernel for counting rows by type (first pass)
__global__ void count_rows_kernel(const int *row_ptr, int n, int threshold, int very_long_threshold,
                                 int *short_count, int *long_count, int *very_long_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ int s_short[256];
    __shared__ int s_long[256];
    __shared__ int s_very_long[256];
    
    int tid = threadIdx.x;
    s_short[tid] = 0;
    s_long[tid] = 0;
    s_very_long[tid] = 0;
    
    if (idx < n) {
        int row_length = row_ptr[idx + 1] - row_ptr[idx];
        if (row_length <= threshold) {
            s_short[tid] = 1;
        } else if (row_length <= very_long_threshold) {
            s_long[tid] = 1;
        } else {
            s_very_long[tid] = 1;
        }
    }
    __syncthreads();
    
    // Block-level reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_short[tid] += s_short[tid + stride];
            s_long[tid] += s_long[tid + stride];
            s_very_long[tid] += s_very_long[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(short_count, s_short[0]);
        atomicAdd(long_count, s_long[0]);
        atomicAdd(very_long_count, s_very_long[0]);
    }
}

void classify_rows_gpu(const int *row_ptr, int n, int **short_rows, int **long_rows, int **very_long_rows,
                      int *num_short, int *num_long, int *num_very_long, int threshold, int very_long_threshold) {
    
    // GPU memory for counters
    int *d_short_count, *d_long_count, *d_very_long_count;
    CUDA_CHECK(cudaMalloc(&d_short_count, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_long_count, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_very_long_count, sizeof(int)));
    
    // Initialize counters
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(d_short_count, &zero, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_long_count, &zero, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_very_long_count, &zero, sizeof(int), cudaMemcpyHostToDevice));
    
    // First pass: count rows
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    count_rows_kernel<<<blocks, threads>>>(row_ptr, n, threshold, very_long_threshold, 
                                          d_short_count, d_long_count, d_very_long_count);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Get counts
    CUDA_CHECK(cudaMemcpy(num_short, d_short_count, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(num_long, d_long_count, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(num_very_long, d_very_long_count, sizeof(int), cudaMemcpyDeviceToHost));
    
    // Allocate arrays on host
    *short_rows = (int*)malloc(*num_short * sizeof(int));
    *long_rows = (int*)malloc(*num_long * sizeof(int));
    *very_long_rows = (int*)malloc(*num_very_long * sizeof(int));
    
    // if (*num_short > 0 && !*short_rows) {
    //     printf("Error: Failed to allocate memory for short rows\n");
    //     return;
    // }
    // if (*num_long > 0 && !*long_rows) {
    //     printf("Error: Failed to allocate memory for long rows\n");
    //     return;
    // }
    // if (*num_very_long > 0 && !*very_long_rows) {
    //     printf("Error: Failed to allocate memory for very long rows\n");
    //     return;
    // }
    
    // GPU arrays for classification
    int *d_short_rows, *d_long_rows, *d_very_long_rows;
    if (*num_short > 0) {
        CUDA_CHECK(cudaMalloc(&d_short_rows, *num_short * sizeof(int)));
    }
    if (*num_long > 0) {
        CUDA_CHECK(cudaMalloc(&d_long_rows, *num_long * sizeof(int)));
    }
    if (*num_very_long > 0) {
        CUDA_CHECK(cudaMalloc(&d_very_long_rows, *num_very_long * sizeof(int)));
    }
    
    // Reset counters for second pass
    CUDA_CHECK(cudaMemcpy(d_short_count, &zero, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_long_count, &zero, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_very_long_count, &zero, sizeof(int), cudaMemcpyHostToDevice));
    
    // Second pass: populate arrays
    classify_rows_kernel<<<blocks, threads>>>(row_ptr, n, threshold, very_long_threshold,
                                             (*num_short > 0) ? d_short_rows : nullptr,
                                             (*num_long > 0) ? d_long_rows : nullptr,
                                             (*num_very_long > 0) ? d_very_long_rows : nullptr,
                                             d_short_count, d_long_count, d_very_long_count);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy results back to host
    if (*num_short > 0) {
        CUDA_CHECK(cudaMemcpy(*short_rows, d_short_rows, *num_short * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_short_rows));
    }
    if (*num_long > 0) {
        CUDA_CHECK(cudaMemcpy(*long_rows, d_long_rows, *num_long * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_long_rows));
    }
    if (*num_very_long > 0) {
        CUDA_CHECK(cudaMemcpy(*very_long_rows, d_very_long_rows, *num_very_long * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_very_long_rows));
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_short_count));
    CUDA_CHECK(cudaFree(d_long_count));
    CUDA_CHECK(cudaFree(d_very_long_count));
}

void classify_rows(const int *row_ptr, int n, int **short_rows, int **long_rows, int **very_long_rows,
                   int *num_short, int *num_long, int *num_very_long, int threshold, int very_long_threshold) {
    classify_rows_gpu(row_ptr, n, short_rows, long_rows, very_long_rows, num_short, num_long, num_very_long, threshold, very_long_threshold);
}

// ============== KERNEL FORWARD DECLARATION ==============

__global__ void hybrid_adaptive_spmv_optimized(const double *csr_values, const int *csr_row_ptr,
                                              const int *csr_col_indices, const double *vec,
                                              double *res, int n, const int *short_rows, 
                                              const int *long_rows, const int *very_long_rows,
                                              int num_short, int num_long, int num_very_long,
                                              int short_blocks, int long_blocks, int block_offset = 0);

// ============== ADAPTIVE LAUNCH CONFIGURATION ==============

struct LaunchConfig {
    int threads_per_block;
    int threshold;
    int short_blocks;
    int long_blocks;
    double performance_score;
};

struct TuningParams {
    int threads;
    int threshold;
    double score;
};

// Benchmark a specific configuration and return performance score
double benchmark_configuration(int threads, int threshold, int very_long_threshold,
                              const double *csr_vals, const int *row_ptr, const int *csr_cols,
                              const double *v, double *C, int rows, int cols, int values) {
    
    // Classify rows with current threshold
    int *short_rows, *long_rows, *very_long_rows;
    int num_short, num_long, num_very_long;
    classify_rows(row_ptr, rows, &short_rows, &long_rows, &very_long_rows, 
                  &num_short, &num_long, &num_very_long, threshold, very_long_threshold);
    
    // Calculate blocks configuration
    int short_blocks_limit = (num_short + threads - 1) / threads;
    int long_blocks = (num_long + (threads / WARP_SIZE) - 1) / (threads / WARP_SIZE);
    int very_long_blocks = num_very_long; // 1 block per very long row
    int regular_blocks = short_blocks_limit + long_blocks;
    
    // Transfer row arrays to GPU
    int *d_short_rows = nullptr, *d_long_rows = nullptr, *d_very_long_rows = nullptr;
    if (num_short > 0) {
        CUDA_CHECK(cudaMalloc(&d_short_rows, num_short * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_short_rows, short_rows, num_short * sizeof(int), cudaMemcpyHostToDevice));
    }
    if (num_long > 0) {
        CUDA_CHECK(cudaMalloc(&d_long_rows, num_long * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_long_rows, long_rows, num_long * sizeof(int), cudaMemcpyHostToDevice));
    }
    if (num_very_long > 0) {
        CUDA_CHECK(cudaMalloc(&d_very_long_rows, num_very_long * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_very_long_rows, very_long_rows, num_very_long * sizeof(int), cudaMemcpyHostToDevice));
    }
    
    // Benchmark with multiple runs
    const int benchmark_iterations = 5;
    double total_time = 0.0;
    
    // Parameters for very long rows
    int very_long_block_size = 1024;
    size_t very_long_shared_mem = very_long_block_size * sizeof(double);
    
    for (int i = 0; i < benchmark_iterations; i++) {
        CUDA_CHECK(cudaMemset(C, 0, rows * sizeof(double)));
        
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
        
        // Launch short and long rows together
        if (regular_blocks > 0) {
            hybrid_adaptive_spmv_optimized<<<regular_blocks, threads>>>(
                csr_vals, row_ptr, csr_cols, v, C, rows, 
                d_short_rows, d_long_rows, d_very_long_rows,
                num_short, num_long, num_very_long, short_blocks_limit, long_blocks, 0);
        }
        
        // Launch very long rows separately with optimal block size
        if (very_long_blocks > 0) {
            hybrid_adaptive_spmv_optimized<<<very_long_blocks, very_long_block_size, very_long_shared_mem>>>(
                csr_vals, row_ptr, csr_cols, v, C, rows, 
                d_short_rows, d_long_rows, d_very_long_rows,
                num_short, num_long, num_very_long, short_blocks_limit, long_blocks, regular_blocks);
        }
        
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float elapsed_time;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        total_time += elapsed_time;
        
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    
    // Cleanup
    if (d_short_rows) CUDA_CHECK(cudaFree(d_short_rows));
    if (d_long_rows) CUDA_CHECK(cudaFree(d_long_rows));
    if (d_very_long_rows) CUDA_CHECK(cudaFree(d_very_long_rows));
    free(short_rows);
    free(long_rows);
    free(very_long_rows);
    
    double avg_time = total_time / benchmark_iterations;
    // Return performance score (higher is better - we use 1/time for maximization)
    return 1000.0 / avg_time;  // Scale factor for better numerical stability
}

// Gradient descent-like parameter tuning
void tune_launch_parameters(int n, int nnz, const int *row_ptr,
                           const double *csr_vals, const int *csr_cols,
                           const double *v, double *C, int rows, int cols, int values,
                           int &best_threads, int &best_threshold, int &best_very_long_threshold) {
    
    // Parameter search spaces
    int thread_options[] = {64, 128, 256, 512};
    int num_thread_options = sizeof(thread_options) / sizeof(thread_options[0]);
    
    // Initialize with matrix analysis for starting point
    struct MAT_STATS stats = calculate_matrix_stats_gpu(row_ptr, n);
    
    // Determine threshold search range based on matrix characteristics
    int min_threshold = 4;
    int max_threshold = std::min(256, (int)(stats.mean_nnz_per_row * 4));
    if (max_threshold < min_threshold) max_threshold = min_threshold * 4;
    
    // Very long threshold options
    int very_long_options[] = {256, 512, 1024, 2048};
    int num_very_long_options = sizeof(very_long_options) / sizeof(very_long_options[0]);
    
    double best_score = 0.0;
    best_threads = 256;  // Default fallback
    best_threshold = 32;
    best_very_long_threshold = 1024;
    
    // Phase 1: Coarse grid search
    for (int t = 0; t < num_thread_options; t++) {
        int threads = thread_options[t];
        
        // Test multiple threshold values
        for (int threshold = min_threshold; threshold <= max_threshold; threshold *= 2) {
            // Test different very long thresholds
            for (int vl = 0; vl < num_very_long_options; vl++) {
                int very_long_threshold = very_long_options[vl];
                if (very_long_threshold <= threshold) continue; // Must be larger than regular threshold
                
                double score = benchmark_configuration(threads, threshold, very_long_threshold, 
                                                     csr_vals, row_ptr, csr_cols, v, C, rows, cols, values);
                
                if (score > best_score) {
                    best_score = score;
                    best_threads = threads;
                    best_threshold = threshold;
                    best_very_long_threshold = very_long_threshold;
                }
            }
        }
    }
    
    // Phase 2: Fine-tuning around best configuration
    
    // Fine-tune threshold around best value
    int threshold_start = std::max(min_threshold, best_threshold / 2);
    int threshold_end = std::min(max_threshold, best_threshold * 2);
    int threshold_step = std::max(1, (threshold_end - threshold_start) / 8);
    
    for (int threshold = threshold_start; threshold <= threshold_end; threshold += threshold_step) {
        if (threshold == best_threshold) continue; // Already tested
        
        double score = benchmark_configuration(best_threads, threshold, best_very_long_threshold, 
                                             csr_vals, row_ptr, csr_cols, v, C, rows, cols, values);
        
        if (score > best_score) {
            best_score = score;
            best_threshold = threshold;
        }
    }
}

void get_hybrid_launch_config(int n, int nnz, const int *row_ptr, 
                             int &blocks, int &threads, 
                             int **short_rows, int **long_rows, int **very_long_rows,
                             int *num_short, int *num_long, int *num_very_long, 
                             int *short_blocks_limit, int *long_blocks_limit,
                             const double *csr_vals, const int *csr_cols,
                             const double *v, double *C, int rows, int cols, int values) {
    
    int optimal_threads, optimal_threshold, optimal_very_long_threshold;
    
    double time;
    TIMER_DEF(var);
    TIMER_START(var);                           

    // Run parameter tuning
    tune_launch_parameters(n, nnz, row_ptr, csr_vals, csr_cols, v, C, rows, cols, values,
                          optimal_threads, optimal_threshold, optimal_very_long_threshold);
    
    // Apply optimal configuration
    threads = optimal_threads;

    TIMER_STOP(var);
    time = TIMER_ELAPSED(var) / 1000.0; // Convert to ms
    printf("Tuning time: %.3f ms\n", time);

    TIMER_DEF(var2);
    TIMER_START(var2);
    
    classify_rows(row_ptr, n, short_rows, long_rows, very_long_rows, 
                  num_short, num_long, num_very_long, optimal_threshold, optimal_very_long_threshold);
    
    TIMER_STOP(var2);
    time = TIMER_ELAPSED(var2) / 1000.0; // Convert to ms
    printf("Preprocessing time: %.3f ms\n", time);

    // printf("\nFinal configuration: THREADS=%d THRESHOLD=%d VERY_LONG_THRESHOLD=%d\n", 
    //        threads, optimal_threshold, optimal_very_long_threshold);
    // printf("Row classification (threshold=%d, very_long_threshold=%d):\n", 
    //        optimal_threshold, optimal_very_long_threshold);
    // printf("  Short rows: %d (%.1f%%)\n", *num_short, 100.0 * (*num_short) / n);
    // printf("  Long rows: %d (%.1f%%)\n", *num_long, 100.0 * (*num_long) / n);
    // printf("  Very long rows: %d (%.1f%%)\n", *num_very_long, 100.0 * (*num_very_long) / n);
    
    // Calculate launch configuration
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
                                              int short_blocks, int long_blocks, int block_offset) {
    
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
        
        // Vectorized computation with manual unrolling
        int element_idx = row_start_idx;
        while (element_idx + 3 < row_end_idx) {
            accumulator += csr_values[element_idx]     * __ldg(&vec[csr_col_indices[element_idx]]);
            accumulator += csr_values[element_idx + 1] * __ldg(&vec[csr_col_indices[element_idx + 1]]);
            accumulator += csr_values[element_idx + 2] * __ldg(&vec[csr_col_indices[element_idx + 2]]);
            accumulator += csr_values[element_idx + 3] * __ldg(&vec[csr_col_indices[element_idx + 3]]);
            element_idx += 4;
        }
        
        // Process remaining elements sequentially
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
        
        // Parallel reduction across warp with strided access
        for (int idx = row_begin + warp_lane; idx < row_finish; idx += WARP_SIZE) {
            partial_result += csr_values[idx] * __ldg(&vec[csr_col_indices[idx]]);
        }
        
        // Butterfly reduction within warp
        for (int stride = WARP_SIZE / 2; stride > 0; stride /= 2) {
            partial_result += __shfl_down_sync(0xFFFFFFFF, partial_result, stride);
        }
        
        // First thread in warp writes final result
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

// GPU kernel for counting entries per row
__global__ void count_row_entries_kernel(const int *rows, int num_entries, int *row_counts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_entries) {
        atomicAdd(&row_counts[rows[idx]], 1);
    }
}

// GPU kernel for prefix sum (simple implementation for small arrays)
__global__ void prefix_sum_kernel(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Simple sequential prefix sum - works for small arrays
    if (idx == 0) {
        for (int i = 1; i <= n; i++) {
            data[i] += data[i - 1];
        }
    }
}

// GPU kernel for filling CSR arrays
__global__ void fill_csr_kernel(const int *rows, const int *cols, const double *vals,
                               int num_entries, const int *row_ptr, 
                               int *csr_cols, double *csr_vals, int *temp_ptr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_entries) {
        int row = rows[idx];
        int pos = atomicAdd(&temp_ptr[row], 1);
        csr_cols[pos] = cols[idx];
        csr_vals[pos] = vals[idx];
    }
}

void coo_to_csr_gpu(int *Arows, int *Acols, double *Avals, 
                    int **row_ptr, int **csr_cols, double **csr_vals,
                    int rows, int values) {
    
    CUDA_CHECK(cudaMallocManaged(row_ptr, (rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(csr_cols, values * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(csr_vals, values * sizeof(double)));
    
    // Initialize row_ptr on GPU
    CUDA_CHECK(cudaMemset(*row_ptr, 0, (rows + 1) * sizeof(int)));
    
    // Count entries per row on GPU
    int threads = 256;
    int blocks = (values + threads - 1) / threads;
    count_row_entries_kernel<<<blocks, threads>>>(Arows, values, *row_ptr + 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Compute prefix sum on GPU
    prefix_sum_kernel<<<1, 1>>>(*row_ptr, rows);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Create temporary row pointers for filling
    int *temp_ptr;
    CUDA_CHECK(cudaMalloc(&temp_ptr, (rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(temp_ptr, *row_ptr, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    
    // Fill CSR arrays on GPU
    fill_csr_kernel<<<blocks, threads>>>(Arows, Acols, Avals, values, *row_ptr, 
                                        *csr_cols, *csr_vals, temp_ptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Cleanup
    CUDA_CHECK(cudaFree(temp_ptr));
}

void coo_to_csr(int *Arows, int *Acols, double *Avals, 
                int **row_ptr, int **csr_cols, double **csr_vals,
                int rows, int values) {
    coo_to_csr_gpu(Arows, Acols, Avals, row_ptr, csr_cols, csr_vals, rows, values);
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
    
    // Read matrix file
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
                
                Arows[counter] = row - 1;  // Convert to 0-based
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
    
    // Convert to CSR
    int *row_ptr, *csr_cols;
    double *csr_vals;
    coo_to_csr(Arows, Acols, Avals, &row_ptr, &csr_cols, &csr_vals, rows, values);
    
    // Create vectors
    double *v, *C;
    CUDA_CHECK(cudaMallocManaged(&v, cols * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&C, rows * sizeof(double)));
    
    // Initialize input vector
    for (int i = 0; i < cols; i++) {
        v[i] = 1.0;
    }
    
    // Get adaptive launch configuration
    int blocks, threads;
    int *short_rows, *long_rows, *very_long_rows;
    int num_short, num_long, num_very_long, short_blocks_limit, long_blocks_limit;
    
    get_hybrid_launch_config(rows, values, row_ptr, blocks, threads,
                            &short_rows, &long_rows, &very_long_rows, 
                            &num_short, &num_long, &num_very_long, 
                            &short_blocks_limit, &long_blocks_limit,
                            csr_vals, csr_cols, v, C, rows, cols, values);
    
    // Transfer row arrays to GPU
    int *d_short_rows = nullptr, *d_long_rows = nullptr, *d_very_long_rows = nullptr;
    if (num_short > 0) {
        CUDA_CHECK(cudaMalloc(&d_short_rows, num_short * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_short_rows, short_rows, num_short * sizeof(int), cudaMemcpyHostToDevice));
    }
    if (num_long > 0) {
        CUDA_CHECK(cudaMalloc(&d_long_rows, num_long * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_long_rows, long_rows, num_long * sizeof(int), cudaMemcpyHostToDevice));
    }
    if (num_very_long > 0) {
        CUDA_CHECK(cudaMalloc(&d_very_long_rows, num_very_long * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_very_long_rows, very_long_rows, num_very_long * sizeof(int), cudaMemcpyHostToDevice));
    }
    
    // Warmup and timing
    cudaEvent_t start, stop;
    double totalTime = 0.0;
    
    // Parameters for very long rows
    int very_long_block_size = 1024;
    size_t very_long_shared_mem = very_long_block_size * sizeof(double);
    int very_long_blocks = num_very_long;
    int regular_blocks = short_blocks_limit + long_blocks_limit;
    
    for (int i = 0; i < ITERATIONS; i++) {
        CUDA_CHECK(cudaMemset(C, 0, rows * sizeof(double)));
        
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
        
        // Launch short and long rows together
        if (regular_blocks > 0) {
            hybrid_adaptive_spmv_optimized<<<regular_blocks, threads>>>(
                csr_vals, row_ptr, csr_cols, v, C, rows, 
                d_short_rows, d_long_rows, d_very_long_rows,
                num_short, num_long, num_very_long, short_blocks_limit, long_blocks_limit, 0);
            checkCudaError("hybrid_adaptive_spmv_optimized (short/long)");
        }
        
        // Launch very long rows separately with optimal block size and correct offset
        if (very_long_blocks > 0) {
            hybrid_adaptive_spmv_optimized<<<very_long_blocks, very_long_block_size, very_long_shared_mem>>>(
                csr_vals, row_ptr, csr_cols, v, C, rows, 
                d_short_rows, d_long_rows, d_very_long_rows,
                num_short, num_long, num_very_long, short_blocks_limit, long_blocks_limit, regular_blocks);
            checkCudaError("hybrid_adaptive_spmv_optimized (very long)");
        }
        
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float elapsed_time;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        
        if (i > 0) {  // Skip first iteration for warmup
            totalTime += elapsed_time;
        }
        
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
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
    
    // Print result vector
    // printf("\nResult vector (one value per line):\n");
    // print_result_vector(C, rows);
    
    // Cleanup
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
    if (d_very_long_rows) CUDA_CHECK(cudaFree(d_very_long_rows));
    
    free(short_rows);
    free(long_rows);
    free(very_long_rows);
    
    return 0;
}