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

// ============== GPU MATRIX STATISTICS ==============

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

void classify_rows(const int *row_ptr, int n, int **short_rows, int **long_rows, 
                   int *num_short, int *num_long, int threshold) {
    
    // First pass: count short and long rows
    *num_short = 0;
    *num_long = 0;
    
    for (int i = 0; i < n; i++) {
        int row_length = row_ptr[i + 1] - row_ptr[i];
        if (row_length <= threshold) {
            (*num_short)++;
        } else {
            (*num_long)++;
        }
    }
    
    // Allocate arrays
    *short_rows = (int*)malloc(*num_short * sizeof(int));
    *long_rows = (int*)malloc(*num_long * sizeof(int));
    
    if (!*short_rows || !*long_rows) {
        printf("Error: Failed to allocate memory for row classification\n");
        return;
    }
    
    // Second pass: populate arrays
    int short_idx = 0, long_idx = 0;
    for (int i = 0; i < n; i++) {
        int row_length = row_ptr[i + 1] - row_ptr[i];
        if (row_length <= threshold) {
            (*short_rows)[short_idx++] = i;
        } else {
            (*long_rows)[long_idx++] = i;
        }
    }
}

// ============== GPU ROW CLASSIFICATION ==============

// GPU kernel for row classification
__global__ void classify_rows_kernel(const int *row_ptr, int n, int threshold,
                                   int *short_rows, int *long_rows,
                                   int *short_count, int *long_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        int row_length = row_ptr[idx + 1] - row_ptr[idx];
        
        if (row_length <= threshold) {
            int pos = atomicAdd(short_count, 1);
            short_rows[pos] = idx;
        } else {
            int pos = atomicAdd(long_count, 1);
            long_rows[pos] = idx;
        }
    }
}

// GPU kernel for counting rows by type (first pass)
__global__ void count_rows_kernel(const int *row_ptr, int n, int threshold,
                                 int *short_count, int *long_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ int s_short[256];
    __shared__ int s_long[256];
    
    int tid = threadIdx.x;
    s_short[tid] = 0;
    s_long[tid] = 0;
    
    if (idx < n) {
        int row_length = row_ptr[idx + 1] - row_ptr[idx];
        if (row_length <= threshold) {
            s_short[tid] = 1;
        } else {
            s_long[tid] = 1;
        }
    }
    __syncthreads();
    
    // Block-level reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_short[tid] += s_short[tid + stride];
            s_long[tid] += s_long[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(short_count, s_short[0]);
        atomicAdd(long_count, s_long[0]);
    }
}

void classify_rows_gpu(const int *row_ptr, int n, int **short_rows, int **long_rows, 
                      int *num_short, int *num_long, int threshold) {
    
    // GPU memory for counters
    int *d_short_count, *d_long_count;
    CUDA_CHECK(cudaMalloc(&d_short_count, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_long_count, sizeof(int)));
    
    // Initialize counters
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(d_short_count, &zero, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_long_count, &zero, sizeof(int), cudaMemcpyHostToDevice));
    
    // First pass: count rows
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    count_rows_kernel<<<blocks, threads>>>(row_ptr, n, threshold, d_short_count, d_long_count);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Get counts
    CUDA_CHECK(cudaMemcpy(num_short, d_short_count, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(num_long, d_long_count, sizeof(int), cudaMemcpyDeviceToHost));
    
    // Allocate arrays on host
    *short_rows = (int*)malloc(*num_short * sizeof(int));
    *long_rows = (int*)malloc(*num_long * sizeof(int));
    
    if (*num_short > 0 && !*short_rows) {
        printf("Error: Failed to allocate memory for short rows\n");
        return;
    }
    if (*num_long > 0 && !*long_rows) {
        printf("Error: Failed to allocate memory for long rows\n");
        return;
    }
    
    // GPU arrays for classification
    int *d_short_rows, *d_long_rows;
    if (*num_short > 0) {
        CUDA_CHECK(cudaMalloc(&d_short_rows, *num_short * sizeof(int)));
    }
    if (*num_long > 0) {
        CUDA_CHECK(cudaMalloc(&d_long_rows, *num_long * sizeof(int)));
    }
    
    // Reset counters for second pass
    CUDA_CHECK(cudaMemcpy(d_short_count, &zero, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_long_count, &zero, sizeof(int), cudaMemcpyHostToDevice));
    
    // Second pass: populate arrays
    classify_rows_kernel<<<blocks, threads>>>(row_ptr, n, threshold,
                                             (*num_short > 0) ? d_short_rows : nullptr,
                                             (*num_long > 0) ? d_long_rows : nullptr,
                                             d_short_count, d_long_count);
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
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_short_count));
    CUDA_CHECK(cudaFree(d_long_count));
}

// ============== KERNEL FORWARD DECLARATION ==============

__global__ void hybrid_adaptive_spmv_optimized(const double *csr_values, const int *csr_row_ptr,
                                              const int *csr_col_indices, const double *vec,
                                              double *res, int n, const int *short_rows, 
                                              const int *long_rows, int num_short, 
                                              int num_long, int short_blocks);

struct PreprocessingTimes {
    double cpu_time;
    double gpu_time;
};

struct PreprocessingTimes preprocessing_benchmark(int n, int nnz, const int *row_ptr) {
    
    int optimal_threshold = 32; // Default value
    int *short_rows_cpu, *long_rows_cpu, *short_rows_gpu, *long_rows_gpu;
    int num_short_cpu, num_long_cpu, num_short_gpu, num_long_gpu;
    
    // Benchmark CPU preprocessing
    TIMER_DEF(cpu_timer);
    TIMER_START(cpu_timer);
    
    // CPU matrix statistics
    struct CSR temp_csr;
    temp_csr.row_pointers = (int*)row_ptr;
    temp_csr.num_rows = n;
    temp_csr.num_non_zeros = nnz;
    struct MAT_STATS stats_cpu = calculate_matrix_stats(&temp_csr);
    
    // CPU row classification
    classify_rows(row_ptr, n, &short_rows_cpu, &long_rows_cpu, &num_short_cpu, &num_long_cpu, optimal_threshold);
    
    TIMER_STOP(cpu_timer);
    double cpu_time = TIMER_ELAPSED(cpu_timer) / 1000.0; // Convert to ms
    printf("CPU preprocessing time: %.3f ms\n", cpu_time);
    
    // Benchmark GPU preprocessing
    TIMER_DEF(gpu_timer);
    TIMER_START(gpu_timer);
    
    // GPU matrix statistics
    struct MAT_STATS stats_gpu = calculate_matrix_stats_gpu(row_ptr, n);
    
    // GPU row classification
    classify_rows_gpu(row_ptr, n, &short_rows_gpu, &long_rows_gpu, &num_short_gpu, &num_long_gpu, optimal_threshold);
    
    TIMER_STOP(gpu_timer);
    double gpu_time = TIMER_ELAPSED(gpu_timer) / 1000.0; // Convert to ms
    printf("GPU preprocessing time: %.3f ms\n", gpu_time);
    
    // Cleanup
    if (short_rows_cpu) free(short_rows_cpu);
    if (long_rows_cpu) free(long_rows_cpu);
    if (short_rows_gpu) free(short_rows_gpu);
    if (long_rows_gpu) free(long_rows_gpu);
    
    struct PreprocessingTimes times;
    times.cpu_time = cpu_time;
    times.gpu_time = gpu_time;
    return times;
}

// ============== DUMMY KERNEL FOR BENCHMARKING ==============

// Dual-strategy SpMV kernel with row-based partitioning
__global__ void hybrid_adaptive_spmv_optimized(const double *csr_values, const int *csr_row_ptr,
                                              const int *csr_col_indices, const double *vec,
                                              double *res, int n, const int *short_rows, 
                                              const int *long_rows, int num_short, 
                                              int num_long, int short_blocks) {
    
    const int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_lane = global_thread_id & (WARP_SIZE - 1);
    
    // Strategy A: Thread-level processing for sparse rows
    if (blockIdx.x < short_blocks) {
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
    else {
        const int warps_per_block = blockDim.x / WARP_SIZE;
        const int block_warp_id = threadIdx.x / WARP_SIZE;
        const int global_warp_id = (blockIdx.x - short_blocks) * warps_per_block + block_warp_id;
        
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

// ============== MAIN FUNCTION ==============

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
        return 1;
    }
    
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
    
    // Convert COO to CSR (CPU only)
    int *row_ptr, *csr_cols;
    double *csr_vals;
    
    coo_to_csr(Arows, Acols, Avals, &row_ptr, &csr_cols, &csr_vals, rows, values);
    
    // Run preprocessing benchmark
    struct PreprocessingTimes prep_times = preprocessing_benchmark(rows, values, row_ptr);
    
    // Print only preprocessing times
    // Note: COO conversion time is not measured or reported
    
    // Cleanup
    CUDA_CHECK(cudaFree(Arows));
    CUDA_CHECK(cudaFree(Acols));
    CUDA_CHECK(cudaFree(Avals));
    CUDA_CHECK(cudaFree(row_ptr));
    CUDA_CHECK(cudaFree(csr_cols));
    CUDA_CHECK(cudaFree(csr_vals));
    
    return 0;
}
