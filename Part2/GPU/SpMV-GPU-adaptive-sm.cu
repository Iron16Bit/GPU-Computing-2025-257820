#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <errno.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include "my_time_lib.h"

// ============== CONFIGURABLE PARAMETERS ==============
#define DEFAULT_THREADS_PER_BLOCK 256
#define SHORT_THRESHOLD 2
#define MEDIUM_THRESHOLD 32
#define LONG_THRESHOLD 128

void print_double_array(double* a, int n) {
    for (int i=0; i<n; i++) {
        printf("%.6f\n", a[i]);
    }
}

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

// Helper function for checking CUDA errors
void checkCudaError(const char* msg) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error after %s: %s\n", msg, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

// Default thresholds - configurable at compile time
struct ThresholdConfig {
    int short_threshold;
    int medium_threshold;
    int long_threshold;
};

// Initialize with compile-time defaults
ThresholdConfig get_default_config() {
    return {SHORT_THRESHOLD, MEDIUM_THRESHOLD, LONG_THRESHOLD};
}

// Row type classifications with contiguous arrays
struct RowClassification {
    int *row_indices;      // All row indices in order: [short_rows][medium_rows][long_rows][very_long_rows]
    int *row_types;        // Row type for each entry: 0=short, 1=medium, 2=long, 3=very_long
    int num_short;
    int num_medium;
    int num_long;
    int num_very_long;
    int total_rows;        // Total classified rows
};

// ============== SINGLE OPTIMIZED HYBRID KERNEL ==============

__global__ void spmv_hybrid_kernel(int *row_ptr, int *Acols, double *Avals, double *v, double *C,
                                  int *row_indices, int *row_types, int total_rows,
                                  int short_threshold, int medium_threshold, int long_threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = idx / 32;
    int lane_id = idx & 31;
    int tid = threadIdx.x;
    
    // Shared memory for cooperative operations
    extern __shared__ double shared_mem[];
    
    // Process regular rows (short, medium, long) - one thread per row
    if (idx < total_rows) {
        int row = row_indices[idx];
        int row_type = row_types[idx];
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];
        int row_length = row_end - row_start;
        
        if (row_type == 0) {
            // Short rows (0-2 elements) - direct computation
            double sum = 0.0;
            for (int j = row_start; j < row_end; j++) {
                sum += Avals[j] * v[Acols[j]];
            }
            C[row] = sum;
            
        } else if (row_type == 1) {
            // Medium rows (3-32 elements) - unrolled computation
            double sum = 0.0;
            int j = row_start;
            // Unroll by 4 for better ILP
            for (; j + 3 < row_end; j += 4) {
                sum += Avals[j] * v[Acols[j]] +
                       Avals[j + 1] * v[Acols[j + 1]] +
                       Avals[j + 2] * v[Acols[j + 2]] +
                       Avals[j + 3] * v[Acols[j + 3]];
            }
            // Handle remaining elements
            for (; j < row_end; j++) {
                sum += Avals[j] * v[Acols[j]];
            }
            C[row] = sum;
            
        } else if (row_type == 2) {
            // Long rows (33-128 elements) - chunked processing with shared memory
            double sum = 0.0;
            int warp_start = (tid / 32) * 32;
            int warp_offset = tid - warp_start;
            
            // Use shared memory to cache vector values in chunks
            const int CHUNK_SIZE = 32;
            for (int chunk_start = 0; chunk_start < row_length; chunk_start += CHUNK_SIZE) {
                int chunk_end = min(chunk_start + CHUNK_SIZE, row_length);
                
                // Load vector values cooperatively within warp
                if (warp_offset < (chunk_end - chunk_start)) {
                    int col_idx = Acols[row_start + chunk_start + warp_offset];
                    shared_mem[warp_start + warp_offset] = v[col_idx];
                }
                __syncwarp(0xffffffff);
                
                // Compute using cached values
                for (int j = chunk_start; j < chunk_end; j++) {
                    sum += Avals[row_start + j] * shared_mem[warp_start + (j - chunk_start)];
                }
                __syncwarp(0xffffffff);
            }
            C[row] = sum;
        }
    }
    
    // Handle very long rows with warp cooperation
    // Process very long rows using warp-level parallelism
    int very_long_warp_id = warp_id;
    if (very_long_warp_id < total_rows && row_types && row_types[very_long_warp_id] == 3) {
        int row = row_indices[very_long_warp_id];
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];
        
        double sum = 0.0;
        // Each thread in warp processes every 32nd element
        for (int j = row_start + lane_id; j < row_end; j += 32) {
            sum += Avals[j] * v[Acols[j]];
        }
        
        // Warp shuffle reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        // First thread in warp writes result
        if (lane_id == 0) {
            C[row] = sum;
        }
    }
    
    // Alternative approach: Dynamic row type detection (if classification overhead is too high)
    // This section can be used instead of pre-classification
    /*
    if (idx < total_rows) {
        int row = idx;  // Direct row mapping
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];
        int row_length = row_end - row_start;
        
        if (row_length <= short_threshold) {
            // Process as short row
            double sum = 0.0;
            for (int j = row_start; j < row_end; j++) {
                sum += Avals[j] * v[Acols[j]];
            }
            C[row] = sum;
        } else if (row_length <= medium_threshold) {
            // Process as medium row
            double sum = 0.0;
            int j = row_start;
            for (; j + 3 < row_end; j += 4) {
                sum += Avals[j] * v[Acols[j]] +
                       Avals[j + 1] * v[Acols[j + 1]] +
                       Avals[j + 2] * v[Acols[j + 2]] +
                       Avals[j + 3] * v[Acols[j + 3]];
            }
            for (; j < row_end; j++) {
                sum += Avals[j] * v[Acols[j]];
            }
            C[row] = sum;
        } else {
            // Use warp cooperation for long rows
            double sum = 0.0;
            for (int j = row_start + lane_id; j < row_end; j += 32) {
                sum += Avals[j] * v[Acols[j]];
            }
            
            // Warp reduction
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }
            
            if (lane_id == 0) {
                C[row] = sum;
            }
        }
    }
    */
}

// ============== MATRIX ANALYSIS AND ROW CLASSIFICATION ==============

struct MatrixStats {
    int short_rows;
    int medium_rows; 
    int long_rows;
    int very_long_rows;
    double avg_nnz_per_row;
    int max_nnz_per_row;
    int total_nnz;
};

// ============== MATRIX CONVERSION AND ANALYSIS ==============

void coo_to_csr_optimized(int *Arows, int *Acols, double *Avals, 
                         int **row_ptr, int **csr_cols, double **csr_vals, int **row_lengths,
                         int rows, int values) {
    
    if (!Arows || !Acols || !Avals || !row_ptr || !csr_cols || !csr_vals || !row_lengths) {
        fprintf(stderr, "Error: NULL pointer passed to coo_to_csr_optimized\n");
        exit(EXIT_FAILURE);
    }
    
    if (rows <= 0 || values <= 0) {
        fprintf(stderr, "Error: Invalid matrix dimensions (rows=%d, values=%d)\n", rows, values);
        exit(EXIT_FAILURE);
    }
    
    CUDA_CHECK(cudaMallocManaged(row_ptr, (rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(csr_cols, values * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(csr_vals, values * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(row_lengths, rows * sizeof(int)));
    
    // Initialize
    memset(*row_ptr, 0, (rows + 1) * sizeof(int));
    
    for (int i = 0; i < values; i++) {
        if (Arows[i] < 0 || Arows[i] >= rows) {
            fprintf(stderr, "Error: Invalid row index %d at position %d (matrix has %d rows)\n", 
                    Arows[i], i, rows);
            exit(EXIT_FAILURE);
        }
        (*row_ptr)[Arows[i] + 1]++;
    }
    
    // Prefix sum
    for (int i = 1; i <= rows; i++) {
        (*row_ptr)[i] += (*row_ptr)[i - 1];
    }
    
    // Calculate row lengths
    for (int i = 0; i < rows; i++) {
        (*row_lengths)[i] = (*row_ptr)[i + 1] - (*row_ptr)[i];
    }
    
    // Fill CSR format - use temporary counters to avoid overwriting
    std::vector<int> temp_ptr(rows + 1);
    for (int i = 0; i <= rows; i++) {
        temp_ptr[i] = (*row_ptr)[i];
    }
    
    for (int i = 0; i < values; i++) {
        int row = Arows[i];
        if (Acols[i] < 0 || Acols[i] >= rows) {
            fprintf(stderr, "Error: Invalid column index %d at position %d\n", Acols[i], i);
            exit(EXIT_FAILURE);
        }
        int pos = temp_ptr[row]++;
        (*csr_cols)[pos] = Acols[i];
        (*csr_vals)[pos] = Avals[i];
    }
}

void compute_performance_metrics(int rows, int cols, int values, double time_ms, 
                               const MatrixStats& stats, const RowClassification& classification) {
    // Memory access calculation
    size_t matrix_bytes = sizeof(int) * (rows + 1) + sizeof(int) * values + sizeof(double) * values;
    size_t vector_bytes = sizeof(double) * cols;
    size_t result_bytes = sizeof(double) * rows;
    size_t classification_bytes = sizeof(int) * classification.total_rows * 2; // row_indices + row_types
    
    size_t total_bytes = matrix_bytes + vector_bytes + result_bytes + classification_bytes;
    
    double bandwidth_gb_s = (total_bytes / (1024.0 * 1024.0 * 1024.0)) / (time_ms / 1000.0);
    double gflops = (2.0 * values) / (time_ms * 1.0e6);
    
    printf("Matrix stats:\n");
    printf("  Short rows (≤%d): %d\n", SHORT_THRESHOLD, stats.short_rows);
    printf("  Medium rows (%d-%d): %d\n", SHORT_THRESHOLD + 1, MEDIUM_THRESHOLD, stats.medium_rows);
    printf("  Long rows (%d-%d): %d\n", MEDIUM_THRESHOLD + 1, LONG_THRESHOLD, stats.long_rows);
    printf("  Very long rows (>%d): %d\n", LONG_THRESHOLD, stats.very_long_rows);
    printf("  Avg NNZ per row: %.2f\n", stats.avg_nnz_per_row);
    printf("  Max NNZ per row: %d\n\n", stats.max_nnz_per_row);
    
    printf("Performance:\n");
    printf("  Execution time: %.3f ms\n", time_ms);
    printf("  Bandwidth: %.2f GB/s\n", bandwidth_gb_s);
    printf("  GFLOPS: %.2f\n", gflops);
}

#define ITERATIONS 51

// ============== GPU KERNELS FOR MATRIX ANALYSIS ==============

// GPU kernel to analyze matrix statistics
__global__ void analyze_matrix_kernel(int *row_lengths, int rows, 
                                    int short_threshold, int medium_threshold, int long_threshold,
                                    int *short_count, int *medium_count, int *long_count, 
                                    int *very_long_count, int *total_nnz, int *max_nnz) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Shared memory for block-level reductions
    __shared__ int s_short[DEFAULT_THREADS_PER_BLOCK];
    __shared__ int s_medium[DEFAULT_THREADS_PER_BLOCK];
    __shared__ int s_long[DEFAULT_THREADS_PER_BLOCK];
    __shared__ int s_very_long[DEFAULT_THREADS_PER_BLOCK];
    __shared__ int s_total_nnz[DEFAULT_THREADS_PER_BLOCK];
    __shared__ int s_max_nnz[DEFAULT_THREADS_PER_BLOCK];
    
    int tid = threadIdx.x;
    
    // Initialize shared memory
    s_short[tid] = 0;
    s_medium[tid] = 0;
    s_long[tid] = 0;
    s_very_long[tid] = 0;
    s_total_nnz[tid] = 0;
    s_max_nnz[tid] = 0;
    
    // Process elements
    if (idx < rows) {
        int len = row_lengths[idx];
        s_total_nnz[tid] = len;
        s_max_nnz[tid] = len;
        
        if (len <= short_threshold) {
            s_short[tid] = 1;
        } else if (len <= medium_threshold) {
            s_medium[tid] = 1;
        } else if (len <= long_threshold) {
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
            s_medium[tid] += s_medium[tid + stride];
            s_long[tid] += s_long[tid + stride];
            s_very_long[tid] += s_very_long[tid + stride];
            s_total_nnz[tid] += s_total_nnz[tid + stride];
            s_max_nnz[tid] = max(s_max_nnz[tid], s_max_nnz[tid + stride]);
        }
        __syncthreads();
    }
    
    // Write block results to global memory
    if (tid == 0) {
        atomicAdd(short_count, s_short[0]);
        atomicAdd(medium_count, s_medium[0]);
        atomicAdd(long_count, s_long[0]);
        atomicAdd(very_long_count, s_very_long[0]);
        atomicAdd(total_nnz, s_total_nnz[0]);
        atomicMax(max_nnz, s_max_nnz[0]);
    }
}

// GPU kernel to classify rows and determine their types
__global__ void classify_rows_kernel(int *row_lengths, int rows,
                                   int short_threshold, int medium_threshold, int long_threshold,
                                   int *row_types, int *row_valid_flags) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < rows) {
        int len = row_lengths[idx];
        row_valid_flags[idx] = 1;  // All rows are valid for processing
        
        if (len <= short_threshold) {
            row_types[idx] = 0;  // short
        } else if (len <= medium_threshold) {
            row_types[idx] = 1;  // medium
        } else if (len <= long_threshold) {
            row_types[idx] = 2;  // long
        } else {
            row_types[idx] = 3;  // very long
        }
    }
}

// GPU kernel to compact and organize classified rows
__global__ void compact_rows_kernel(int *row_types, int *row_valid_flags, int *row_indices_out,
                                  int *row_types_out, int *prefix_sums, int rows) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < rows && row_valid_flags[idx]) {
        int output_pos = prefix_sums[idx];
        row_indices_out[output_pos] = idx;
        row_types_out[output_pos] = row_types[idx];
    }
}

// ============== GPU-BASED MATRIX ANALYSIS FUNCTIONS ==============

MatrixStats analyze_matrix_gpu(int *row_lengths, int rows, const ThresholdConfig& config) {
    if (!row_lengths || rows <= 0) {
        fprintf(stderr, "Error: Invalid parameters to analyze_matrix_gpu\n");
        exit(EXIT_FAILURE);
    }
    
    MatrixStats stats = {0};
    
    // Allocate GPU memory for counters
    int *d_short_count, *d_medium_count, *d_long_count, *d_very_long_count;
    int *d_total_nnz, *d_max_nnz;
    
    CUDA_CHECK(cudaMalloc(&d_short_count, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_medium_count, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_long_count, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_very_long_count, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_total_nnz, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_max_nnz, sizeof(int)));
    
    // Initialize counters to zero
    CUDA_CHECK(cudaMemset(d_short_count, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_medium_count, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_long_count, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_very_long_count, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_total_nnz, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_max_nnz, 0, sizeof(int)));
    
    // Launch kernel
    int threads = DEFAULT_THREADS_PER_BLOCK;
    int blocks = (rows + threads - 1) / threads;
    
    analyze_matrix_kernel<<<blocks, threads>>>(
        row_lengths, rows, config.short_threshold, config.medium_threshold, config.long_threshold,
        d_short_count, d_medium_count, d_long_count, d_very_long_count, d_total_nnz, d_max_nnz);
    checkCudaError("analyze_matrix_kernel");
    
    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(&stats.short_rows, d_short_count, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&stats.medium_rows, d_medium_count, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&stats.long_rows, d_long_count, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&stats.very_long_rows, d_very_long_count, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&stats.total_nnz, d_total_nnz, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&stats.max_nnz_per_row, d_max_nnz, sizeof(int), cudaMemcpyDeviceToHost));
    
    stats.avg_nnz_per_row = (double)stats.total_nnz / rows;
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_short_count));
    CUDA_CHECK(cudaFree(d_medium_count));
    CUDA_CHECK(cudaFree(d_long_count));
    CUDA_CHECK(cudaFree(d_very_long_count));
    CUDA_CHECK(cudaFree(d_total_nnz));
    CUDA_CHECK(cudaFree(d_max_nnz));
    
    return stats;
}

RowClassification classify_rows_gpu(int *row_lengths, int rows, const ThresholdConfig& config) {
    if (!row_lengths || rows <= 0) {
        fprintf(stderr, "Error: Invalid parameters to classify_rows_gpu\n");
        exit(EXIT_FAILURE);
    }
    
    RowClassification classification = {0};
    
    // Allocate temporary GPU arrays
    int *d_row_types, *d_row_valid_flags;
    CUDA_CHECK(cudaMalloc(&d_row_types, rows * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_row_valid_flags, rows * sizeof(int)));
    
    // Step 1: Classify rows and mark valid ones
    int threads = DEFAULT_THREADS_PER_BLOCK;
    int blocks = (rows + threads - 1) / threads;
    
    classify_rows_kernel<<<blocks, threads>>>(
        row_lengths, rows, config.short_threshold, config.medium_threshold, config.long_threshold,
        d_row_types, d_row_valid_flags);
    checkCudaError("classify_rows_kernel");
    
    // Step 2: Use Thrust to compute prefix sums for compaction
    thrust::device_vector<int> valid_flags(d_row_valid_flags, d_row_valid_flags + rows);
    thrust::device_vector<int> prefix_sums(rows);
    thrust::exclusive_scan(valid_flags.begin(), valid_flags.end(), prefix_sums.begin());
    
    // Get total number of valid rows
    int total_valid = thrust::reduce(valid_flags.begin(), valid_flags.end());
    classification.total_rows = total_valid;
    
    if (total_valid == 0) {
        printf("Warning: No rows to classify\n");
        CUDA_CHECK(cudaFree(d_row_types));
        CUDA_CHECK(cudaFree(d_row_valid_flags));
        return classification;
    }
    
    // Allocate final unified arrays
    CUDA_CHECK(cudaMallocManaged(&classification.row_indices, total_valid * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&classification.row_types, total_valid * sizeof(int)));
    
    // Step 3: Compact the arrays
    compact_rows_kernel<<<blocks, threads>>>(
        d_row_types, d_row_valid_flags, classification.row_indices, classification.row_types,
        thrust::raw_pointer_cast(prefix_sums.data()), rows);
    checkCudaError("compact_rows_kernel");
    
    // Count rows by type by examining the final arrays
    std::vector<int> h_row_types(total_valid);
    CUDA_CHECK(cudaMemcpy(h_row_types.data(), classification.row_types, 
                         total_valid * sizeof(int), cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < total_valid; i++) {
        switch (h_row_types[i]) {
            case 0: classification.num_short++; break;
            case 1: classification.num_medium++; break;
            case 2: classification.num_long++; break;
            case 3: classification.num_very_long++; break;
        }
    }
    
    // Cleanup temporary arrays
    CUDA_CHECK(cudaFree(d_row_types));
    CUDA_CHECK(cudaFree(d_row_valid_flags));
    
    return classification;
}

void free_row_classification(RowClassification &classification) {
    if (classification.row_indices) CUDA_CHECK(cudaFree(classification.row_indices));
    if (classification.row_types) CUDA_CHECK(cudaFree(classification.row_types));
}

int main(int argc, char *argv[]) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "Usage: %s <input_file> [threads_per_block]\n", argv[0]);
        return 1;
    }

    int threadsPerBlock = DEFAULT_THREADS_PER_BLOCK;
    
    // Parse command line arguments
    if (argc >= 3) {
        int user_threads = atoi(argv[2]);
        if (user_threads > 0 && user_threads <= 1024) {
            threadsPerBlock = user_threads;
        }
    }
    
    printf("Matrix: %s\n", argv[1]);
    printf("Threads per block: %d\n", threadsPerBlock);
    printf("Using GPU preprocessing with classification-based hybrid kernel\n\n");

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
                    fprintf(stderr, "Error: Failed to parse matrix dimensions from header\n");
                    fclose(fin);
                    return 1;
                }
                
                if (rows <= 0 || cols <= 0 || values <= 0) {
                    fprintf(stderr, "Error: Invalid matrix dimensions: %d x %d with %d values\n", 
                            rows, cols, values);
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
                    fprintf(stderr, "Error: Failed to parse matrix entry at line %d\n", counter + 1);
                    fclose(fin);
                    return 1;
                }
                
                if (counter >= values) {
                    fprintf(stderr, "Error: More entries than expected (%d)\n", values);
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
    int *row_ptr, *csr_cols, *row_lengths;
    double *csr_vals;
    coo_to_csr_optimized(Arows, Acols, Avals, &row_ptr, &csr_cols, &csr_vals, &row_lengths, rows, values);

    // Create vectors
    double *v, *C;
    CUDA_CHECK(cudaMallocManaged(&v, cols * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&C, rows * sizeof(double)));
    
    // Initialize input vector
    for (int i = 0; i < cols; i++) {
        v[i] = 1.0;
    }

    // Use compile-time configured parameters
    ThresholdConfig config = get_default_config();
    
    // GPU-based matrix analysis and row classification with timing
    cudaEvent_t prep_start, prep_stop;
    CUDA_CHECK(cudaEventCreate(&prep_start));
    CUDA_CHECK(cudaEventCreate(&prep_stop));
    CUDA_CHECK(cudaEventRecord(prep_start));
    
    MatrixStats stats = analyze_matrix_gpu(row_lengths, rows, config);
    RowClassification classification = classify_rows_gpu(row_lengths, rows, config);
    
    CUDA_CHECK(cudaEventRecord(prep_stop));
    CUDA_CHECK(cudaEventSynchronize(prep_stop));
    
    float prep_time;
    CUDA_CHECK(cudaEventElapsedTime(&prep_time, prep_start, prep_stop));
    printf("GPU preprocessing time: %.3f ms\n\n", prep_time);
    
    CUDA_CHECK(cudaEventDestroy(prep_start));
    CUDA_CHECK(cudaEventDestroy(prep_stop));

    // Warmup and timing
    cudaEvent_t start, stop;
    double totalTime = 0.0;
    
    for (int i = 0; i < ITERATIONS; i++) {
        CUDA_CHECK(cudaMemset(C, 0, rows * sizeof(double)));
        
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));

        // Use classification-based hybrid kernel
        if (classification.total_rows > 0) {
            // Calculate blocks needed for both regular processing and warp cooperation
            int max_threads_needed = max(classification.total_rows, 
                                       (classification.num_very_long * 32)); 
            int blocks = (max_threads_needed + threadsPerBlock - 1) / threadsPerBlock;
            
            // Shared memory for cooperative loading (per warp)
            int warps_per_block = (threadsPerBlock + 31) / 32;
            size_t shared_mem_size = warps_per_block * 32 * sizeof(double);
            
            spmv_hybrid_kernel<<<blocks, threadsPerBlock, shared_mem_size>>>(
                row_ptr, csr_cols, csr_vals, v, C,
                classification.row_indices, classification.row_types, classification.total_rows,
                config.short_threshold, config.medium_threshold, config.long_threshold);
        }
        
        checkCudaError("SpMV kernel execution");

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

    print_double_array(C, rows);

    double avg_time = totalTime / (ITERATIONS - 1);
    compute_performance_metrics(rows, cols, values, avg_time, stats, classification);

    // Cleanup
    CUDA_CHECK(cudaFree(Arows));
    CUDA_CHECK(cudaFree(Acols));
    CUDA_CHECK(cudaFree(Avals));
    CUDA_CHECK(cudaFree(row_ptr));
    CUDA_CHECK(cudaFree(csr_cols));
    CUDA_CHECK(cudaFree(csr_vals));
    CUDA_CHECK(cudaFree(row_lengths));
    CUDA_CHECK(cudaFree(v));
    CUDA_CHECK(cudaFree(C));
    
    // Free row classification arrays
    free_row_classification(classification);

    return 0;
}