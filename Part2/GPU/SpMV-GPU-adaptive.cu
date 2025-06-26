#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <errno.h>

// ============== CONFIGURABLE PARAMETERS ==============
#define DEFAULT_THREADS_PER_BLOCK 256
#define SHORT_THRESHOLD 2
#define MEDIUM_THRESHOLD 32
#define LONG_THRESHOLD 128

void print_double_array(double* a, int n) {
    for (int i=0; i<n; i++) {
        printf("%f ", a[i]);
    }
    printf("\n");
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
    int *row_starts;       // Starting CSR indices for each row
    int *row_ends;         // Ending CSR indices for each row  
    int *row_types;        // Row type for each entry: 0=short, 1=medium, 2=long, 3=very_long
    int num_short;
    int num_medium;
    int num_long;
    int num_very_long;
    int total_rows;        // Total classified rows
};

// ============== UNIFIED HYBRID KERNEL ==============

// Single hybrid kernel that processes all row types efficiently
__global__ void spmv_hybrid_kernel(int *row_ptr, int *Acols, double *Avals, double *v, double *C,
                                  int *row_indices, int *row_types, int total_rows) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Handle very long rows with warp cooperation
    int warp_id = idx / 32;
    int lane_id = idx & 31;
    
    if (idx < total_rows) {
        int row = row_indices[idx];
        int row_type = row_types[idx];
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];
        
        if (row_type == 0) {
            // Short rows (0-2 elements) - simple loop
            double sum = 0.0;
            #pragma unroll
            for (int j = row_start; j < row_end; j++) {
                sum += Avals[j] * v[Acols[j]];
            }
            C[row] = sum;
            
        } else if (row_type == 1) {
            // Medium rows (3-32 elements) - unrolled with ILP
            double sum = 0.0;
            int j = row_start;
            
            // Unroll by 4 for instruction-level parallelism
            for (; j + 3 < row_end; j += 4) {
                double val0 = Avals[j] * v[Acols[j]];
                double val1 = Avals[j + 1] * v[Acols[j + 1]];
                double val2 = Avals[j + 2] * v[Acols[j + 2]];
                double val3 = Avals[j + 3] * v[Acols[j + 3]];
                sum += val0 + val1 + val2 + val3;
            }
            
            // Handle remaining elements
            for (; j < row_end; j++) {
                sum += Avals[j] * v[Acols[j]];
            }
            C[row] = sum;
            
        } else if (row_type == 2) {
            // Long rows (33-128 elements) - blocked processing
            double sum = 0.0;
            
            // Process in chunks for better cache behavior
            const int CHUNK_SIZE = 32;
            for (int chunk_start = row_start; chunk_start < row_end; chunk_start += CHUNK_SIZE) {
                int chunk_end = min(chunk_start + CHUNK_SIZE, row_end);
                double chunk_sum = 0.0;
                
                int j = chunk_start;
                // Unroll inner loop
                for (; j + 3 < chunk_end; j += 4) {
                    double val0 = Avals[j] * v[Acols[j]];
                    double val1 = Avals[j + 1] * v[Acols[j + 1]];
                    double val2 = Avals[j + 2] * v[Acols[j + 2]];
                    double val3 = Avals[j + 3] * v[Acols[j + 3]];
                    chunk_sum += val0 + val1 + val2 + val3;
                }
                for (; j < chunk_end; j++) {
                    chunk_sum += Avals[j] * v[Acols[j]];
                }
                sum += chunk_sum;
            }
            C[row] = sum;
        }
    }
    
    // Handle very long rows with warp cooperation  
    if (warp_id < total_rows) {
        int row_idx = warp_id;
        if (row_idx < total_rows && row_types[row_idx] == 3) {
            int row = row_indices[row_idx];
            int row_start = row_ptr[row];
            int row_end = row_ptr[row + 1];
            
            double sum = 0.0;
            
            // Each thread processes every 32nd element
            for (int j = row_start + lane_id; j < row_end; j += 32) {
                sum += Avals[j] * v[Acols[j]];
            }
            
            // Warp shuffle reduction for efficiency
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }
            
            // First thread writes result
            if (lane_id == 0) {
                C[row] = sum;
            }
        }
    }
}

// ============== MATRIX ANALYSIS AND ROW CLASSIFICATION ==============

RowClassification classify_rows(int *row_lengths, int rows, const ThresholdConfig& config) {
    if (!row_lengths || rows <= 0) {
        fprintf(stderr, "Error: Invalid parameters to classify_rows\n");
        exit(EXIT_FAILURE);
    }
    
    RowClassification classification = {0};
    
    // Count rows by type and create unified arrays
    std::vector<int> short_list, medium_list, long_list, very_long_list;
    
    for (int i = 0; i < rows; i++) {
        int len = row_lengths[i];
        
        if (len <= config.short_threshold) {
            short_list.push_back(i);
        } else if (len <= config.medium_threshold) {
            medium_list.push_back(i);
        } else if (len <= config.long_threshold) {
            long_list.push_back(i);
        } else {
            very_long_list.push_back(i);
        }
    }
    
    // Store counts
    classification.num_short = short_list.size();
    classification.num_medium = medium_list.size();
    classification.num_long = long_list.size();
    classification.num_very_long = very_long_list.size();
    classification.total_rows = classification.num_short + classification.num_medium + 
                               classification.num_long + classification.num_very_long;
    
    if (classification.total_rows == 0) {
        printf("Warning: No rows to classify\n");
        return classification;
    }
    
    // Allocate unified arrays
    CUDA_CHECK(cudaMallocManaged(&classification.row_indices, classification.total_rows * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&classification.row_types, classification.total_rows * sizeof(int)));
    
    // Fill unified arrays: [short][medium][long][very_long]
    int offset = 0;
    
    // Short rows (type 0)
    for (int i = 0; i < classification.num_short; i++) {
        classification.row_indices[offset + i] = short_list[i];
        classification.row_types[offset + i] = 0;
    }
    offset += classification.num_short;
    
    // Medium rows (type 1)
    for (int i = 0; i < classification.num_medium; i++) {
        classification.row_indices[offset + i] = medium_list[i];
        classification.row_types[offset + i] = 1;
    }
    offset += classification.num_medium;
    
    // Long rows (type 2)
    for (int i = 0; i < classification.num_long; i++) {
        classification.row_indices[offset + i] = long_list[i];
        classification.row_types[offset + i] = 2;
    }
    offset += classification.num_long;
    
    // Very long rows (type 3)
    for (int i = 0; i < classification.num_very_long; i++) {
        classification.row_indices[offset + i] = very_long_list[i];
        classification.row_types[offset + i] = 3;
    }
    
    return classification;
}

void free_row_classification(RowClassification &classification) {
    if (classification.row_indices) CUDA_CHECK(cudaFree(classification.row_indices));
    if (classification.row_types) CUDA_CHECK(cudaFree(classification.row_types));
}

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

struct MatrixStats {
    int short_rows;
    int medium_rows; 
    int long_rows;
    int very_long_rows;
    double avg_nnz_per_row;
    int max_nnz_per_row;
    int total_nnz;
};

MatrixStats analyze_matrix(int *row_lengths, int rows, const ThresholdConfig& config) {
    if (!row_lengths || rows <= 0) {
        fprintf(stderr, "Error: Invalid parameters to analyze_matrix\n");
        exit(EXIT_FAILURE);
    }
    
    MatrixStats stats = {0};
    
    for (int i = 0; i < rows; i++) {
        int len = row_lengths[i];
        stats.total_nnz += len;
        
        if (len <= config.short_threshold) {
            stats.short_rows++;
        } else if (len <= config.medium_threshold) {
            stats.medium_rows++;
        } else if (len <= config.long_threshold) {
            stats.long_rows++;
        } else {
            stats.very_long_rows++;
        }
        
        if (len > stats.max_nnz_per_row) {
            stats.max_nnz_per_row = len;
        }
    }
    
    stats.avg_nnz_per_row = (double)stats.total_nnz / rows;
    return stats;
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
    
    printf("Execution time: %.3f ms\n", time_ms);
    printf("Bandwidth: %.2f GB/s\n", bandwidth_gb_s);
    printf("GFLOPS: %.2f\n", gflops);
}

#define ITERATIONS 51

int main(int argc, char *argv[]) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "Usage: %s <input_file> [threads_per_block]\n", argv[0]);
        return 1;
    }

    int threadsPerBlock = DEFAULT_THREADS_PER_BLOCK;
    
    // Parse command line arguments
    if (argc == 3) {
        int user_threads = atoi(argv[2]);
        if (user_threads > 0 && user_threads <= 1024) {
            threadsPerBlock = user_threads;
        }
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
    
    // Analyze matrix and classify rows
    MatrixStats stats = analyze_matrix(row_lengths, rows, config);
    
    // Classify rows by type
    RowClassification classification = classify_rows(row_lengths, rows, config);

    // Warmup and timing
    cudaEvent_t start, stop;
    double totalTime = 0.0;
    
    for (int i = 0; i < ITERATIONS; i++) {
        CUDA_CHECK(cudaMemset(C, 0, rows * sizeof(double)));
        
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));

        // Single unified kernel launch for all row types
        if (classification.total_rows > 0) {
            // Calculate blocks needed considering both regular threads and warp cooperation
            int max_threads_needed = max(classification.total_rows, 
                                       classification.num_very_long * 32); // 32 threads per warp for very long rows
            int blocks = (max_threads_needed + threadsPerBlock - 1) / threadsPerBlock;
            
            spmv_hybrid_kernel<<<blocks, threadsPerBlock>>>(
                row_ptr, csr_cols, csr_vals, v, C,
                classification.row_indices, classification.row_types, classification.total_rows);
            checkCudaError("spmv_hybrid_kernel execution");
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

    // print_double_array(C, rows);

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