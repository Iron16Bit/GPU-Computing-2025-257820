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
#define ULTRA_LONG_THRESHOLD 8192
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
    int ultra_long_rows;
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
    stats.ultra_long_rows = 0;
    
    double sum = 0.0;
    double sum_squares = 0.0;
    
    for (int i = 0; i < matrix->num_rows; i++) {
        int row_nnz = matrix->row_pointers[i + 1] - matrix->row_pointers[i];
        
        if (row_nnz == 0) {
            stats.empty_rows++;
        }
        
        if (row_nnz > ULTRA_LONG_THRESHOLD) {
            stats.ultra_long_rows++;
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
                             ROW_CHUNK **ultra_long_chunks,
                             int *num_short, int *num_long, int *num_ultra_long_chunks,
                             int short_threshold, int long_threshold) {
    
    *num_short = 0;
    *num_long = 0;
    *num_ultra_long_chunks = 0;
    
    std::vector<ROW_CHUNK> temp_chunks;
    int ultra_long_rows_count = 0;
    int max_chunks_for_single_row = 0;
    int total_ultra_long_elements = 0;
    
    for (int i = 0; i < n; i++) {
        int row_length = row_ptr[i + 1] - row_ptr[i];
        
        if (row_length <= short_threshold) {
            (*num_short)++;
        } else if (row_length <= long_threshold) {
            (*num_long)++;
        } else {
            ultra_long_rows_count++;
            total_ultra_long_elements += row_length;
            
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
            *num_ultra_long_chunks += chunks_needed;
        }
    }
    
    *short_rows = (int*)malloc(*num_short * sizeof(int));
    *long_rows = (int*)malloc(*num_long * sizeof(int));
    *ultra_long_chunks = (ROW_CHUNK*)malloc(*num_ultra_long_chunks * sizeof(ROW_CHUNK));
    
    if (!*short_rows || !*long_rows || !*ultra_long_chunks) {
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
        (*ultra_long_chunks)[i] = temp_chunks[i];
    }
}

void get_enhanced_launch_config(int n, int nnz, const int *row_ptr,
                               int &total_blocks, int &threads,
                               int **short_rows, int **long_rows, 
                               ROW_CHUNK **ultra_long_chunks,
                               int *num_short, int *num_long, int *num_ultra_long_chunks,
                               int *short_blocks, int *long_blocks, int *ultra_long_blocks) {
    
    struct CSR temp_csr;
    temp_csr.row_pointers = (int*)row_ptr;
    temp_csr.num_rows = n;
    temp_csr.num_non_zeros = nnz;
    struct MAT_STATS stats = calculate_enhanced_matrix_stats(&temp_csr);
    
    int short_threshold, long_threshold;
    
    if (stats.ultra_long_rows > 0) {
        if (stats.mean_nnz_per_row < 2.0) {
            short_threshold = 1;
            long_threshold = 8;
        } else {
            short_threshold = 4;
            long_threshold = 32;
        }
        threads = 1024;
    } else if (stats.mean_nnz_per_row < 2.0) {
        short_threshold = 8;
        long_threshold = 64;
        threads = 1024;
    } else if (stats.mean_nnz_per_row < 10.0) {
        short_threshold = 16;
        long_threshold = 128;
        threads = 1024;
    } else if (n < 100000) {
        short_threshold = 16;
        long_threshold = 128;
        threads = 1024;
    } else {
        short_threshold = 32;
        long_threshold = 256;
        threads = 1024;
    }
    
    classify_and_split_rows(row_ptr, n, short_rows, long_rows, ultra_long_chunks,
                           num_short, num_long, num_ultra_long_chunks,
                           short_threshold, long_threshold);
    
    *short_blocks = (*num_short + threads - 1) / threads;
    *long_blocks = (*num_long + (threads / WARP_SIZE) - 1) / (threads / WARP_SIZE);
    *ultra_long_blocks = *num_ultra_long_chunks;
    
    total_blocks = *short_blocks + *long_blocks + *ultra_long_blocks;
}

__global__ void enhanced_adaptive_spmv(const double *__restrict__ csr_values, 
                                       const int *__restrict__ csr_row_ptr,
                                       const int *__restrict__ csr_col_indices, 
                                       const double *__restrict__ vec,
                                       double *__restrict__ res, int n, 
                                       const int *__restrict__ short_rows,
                                       const int *__restrict__ long_rows, 
                                       const ROW_CHUNK *__restrict__ ultra_long_chunks,
                                       int num_short, int num_long, int num_ultra_long_chunks,
                                       int short_blocks, int long_blocks, 
                                       int phase, int block_offset = 0) {
    
    const int effective_block_id = blockIdx.x + block_offset;
    const int thread_id = threadIdx.x;
    const int warp_lane = thread_id & (WARP_SIZE - 1);
    const int warp_id = thread_id >> 5;
    
    if (phase == 0 && effective_block_id < short_blocks) {
        const int global_thread_id = effective_block_id * blockDim.x + thread_id;
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
    
    else if (phase == 1 && effective_block_id < long_blocks) {
        int warp_global_id = effective_block_id * (blockDim.x >> 5) + warp_id;
        
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
    
    else if (phase == 2 && effective_block_id < num_ultra_long_chunks) {
        const ROW_CHUNK chunk = ultra_long_chunks[effective_block_id];
        const int row = chunk.row_id;
        const int start = chunk.chunk_start;
        const int end = chunk.chunk_end;
        
        double thread_sum = 0.0;
        
        for (int idx = start + thread_id; idx < end; idx += blockDim.x) {
            thread_sum += csr_values[idx] * __ldg(&vec[csr_col_indices[idx]]);
        }
        
        __shared__ double sdata[256];
        if (thread_id < 256) {
            sdata[thread_id] = thread_sum;
        }
        __syncthreads();
        
        for (int stride = min(blockDim.x, 256) >> 1; stride > 0; stride >>= 1) {
            if (thread_id < stride && thread_id + stride < 256) {
                sdata[thread_id] += sdata[thread_id + stride];
            }
            __syncthreads();
        }
        
        if (thread_id == 0) {
            atomicAdd(&res[row], sdata[0]);
        }
    }
}

__global__ void simple_spmv_small_matrix(const double *__restrict__ csr_values, 
                                         const int *__restrict__ csr_row_ptr,
                                         const int *__restrict__ csr_col_indices, 
                                         const double *__restrict__ vec,
                                         double *__restrict__ res, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n) {
        int start = csr_row_ptr[row];
        int end = csr_row_ptr[row + 1];
        
        double sum = 0.0;
        for (int idx = start; idx < end; idx++) {
            sum += csr_values[idx] * vec[csr_col_indices[idx]];
        }
        res[row] = sum;
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
    ROW_CHUNK *ultra_long_chunks;
    int num_short, num_long, num_ultra_long_chunks;
    int short_blocks, long_blocks, ultra_long_blocks;
    
    get_enhanced_launch_config(rows, values, row_ptr, total_blocks, threads,
                              &short_rows, &long_rows, &ultra_long_chunks,
                              &num_short, &num_long, &num_ultra_long_chunks,
                              &short_blocks, &long_blocks, &ultra_long_blocks);
    
    TIMER_STOP(var);
    time = TIMER_ELAPSED(var) / 1000.0;
    printf("Preprocessing time: %.3f ms\n", time);
    
    int *d_short_rows = nullptr, *d_long_rows = nullptr;
    ROW_CHUNK *d_ultra_long_chunks = nullptr;
    
    if (num_short > 0) {
        CUDA_CHECK(cudaMalloc(&d_short_rows, num_short * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_short_rows, short_rows, num_short * sizeof(int), cudaMemcpyHostToDevice));
    }
    if (num_long > 0) {
        CUDA_CHECK(cudaMalloc(&d_long_rows, num_long * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_long_rows, long_rows, num_long * sizeof(int), cudaMemcpyHostToDevice));
    }
    if (num_ultra_long_chunks > 0) {
        CUDA_CHECK(cudaMalloc(&d_ultra_long_chunks, num_ultra_long_chunks * sizeof(ROW_CHUNK)));
        CUDA_CHECK(cudaMemcpy(d_ultra_long_chunks, ultra_long_chunks, num_ultra_long_chunks * sizeof(ROW_CHUNK), cudaMemcpyHostToDevice));
    }
    
    cudaEvent_t start, stop;
    double totalTime = 0.0;
    
    for (int i = 0; i < ITERATIONS; i++) {
        CUDA_CHECK(cudaMemset(C, 0, rows * sizeof(double)));
        
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
        
        if (short_blocks > 0) {
            enhanced_adaptive_spmv<<<short_blocks, threads>>>(
                csr_vals, row_ptr, csr_cols, v, C, rows,
                d_short_rows, d_long_rows, d_ultra_long_chunks,
                num_short, num_long, num_ultra_long_chunks,
                short_blocks, long_blocks, 0, 0);
            checkCudaError("enhanced_adaptive_spmv phase 0");
        }
        
        if (long_blocks > 0) {
            enhanced_adaptive_spmv<<<long_blocks, threads>>>(
                csr_vals, row_ptr, csr_cols, v, C, rows,
                d_short_rows, d_long_rows, d_ultra_long_chunks,
                num_short, num_long, num_ultra_long_chunks,
                short_blocks, long_blocks, 1, 0);
            checkCudaError("enhanced_adaptive_spmv phase 1");
        }
        
        if (ultra_long_blocks > 0) {
            enhanced_adaptive_spmv<<<ultra_long_blocks, 256>>>(
                csr_vals, row_ptr, csr_cols, v, C, rows,
                d_short_rows, d_long_rows, d_ultra_long_chunks,
                num_short, num_long, num_ultra_long_chunks,
                short_blocks, long_blocks, 2, 0);
            checkCudaError("enhanced_adaptive_spmv phase 2");
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
    if (d_ultra_long_chunks) CUDA_CHECK(cudaFree(d_ultra_long_chunks));
    
    free(short_rows);
    free(long_rows);
    free(ultra_long_chunks);
    
    return 0;
}
