#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// OPTIMIZATION 1: Launch bounds for register and occupancy optimization
// 256 threads per block ensures full warp utilization (256/32 = 8 warps)
// 4 blocks per SM limits register usage to maintain high occupancy
__global__ 
__launch_bounds__(256, 4)  // Controls register allocation and blocks per SM
void spmv(int * __restrict__ Arows,           // OPTIMIZATION 2: __restrict__ tells compiler
          int * __restrict__ Acols,           // no pointer aliasing for better optimization
          double * __restrict__ Avals, 
          const double * __restrict__ v,      // OPTIMIZATION 3: const for read-only data
          double * __restrict__ C, 
          int rows, int cols, int values, 
          const int * __restrict__ Aslices) {
    
    // OPTIMIZATION 4: Use const and register for frequently accessed variables
    // const helps compiler optimize, register hints for register allocation
    const int slice_id = blockIdx.x;
    const int tid = threadIdx.x;
    
    // OPTIMIZATION 5: Fast bitwise operations instead of expensive modulo/division
    const int lane = tid & 31;      // Equivalent to tid % 32, but single cycle
    const int warp_id = tid >> 5;   // Equivalent to tid / 32, but single cycle
    
    // OPTIMIZATION 6: Early exit to reduce divergence and unnecessary computation
    if (slice_id >= gridDim.x) return;
    
    // OPTIMIZATION 7: Cache slice bounds in registers to avoid repeated global memory access
    register const int slice_start = Aslices[slice_id];
    register const int slice_end = Aslices[slice_id + 1];
    register const int slice_nnz = slice_end - slice_start;

    // OPTIMIZATION 8: Early exit for invalid slices to reduce wasted work
    if (slice_nnz <= 0 || slice_start < 0 || slice_end > values) return;

    // OPTIMIZATION 9: Shared memory with alignment to avoid bank conflicts
    // Bank conflicts occur when threads access same memory bank simultaneously
    extern __shared__ char shared[];
    double* __restrict__ sdata = (double*)shared;
    // OPTIMIZATION 10: Align int array to 32-element boundary to prevent bank conflicts
    int* __restrict__ srows = (int*)&sdata[(slice_nnz + 32) & ~31];

    // OPTIMIZATION 11: Coalesced memory loading with loop unrolling
    // #pragma unroll reduces loop overhead and enables better instruction scheduling
    #pragma unroll 4
    for (int i = tid; i < slice_nnz; i += blockDim.x) {
        register int idx = slice_start + i;  // OPTIMIZATION 12: register hint for hot variables
        if (idx < values) {
            // OPTIMIZATION 13: __ldg() uses read-only texture cache for better bandwidth
            // This avoids polluting L1 cache with data that won't be reused
            register int col = __ldg(&Acols[idx]);
            if (col >= 0 && col < cols) {
                srows[i] = __ldg(&Arows[idx]);
                // OPTIMIZATION 14: Multiply directly during load to save memory access later
                // Combines matrix value with vector element in one operation
                sdata[i] = __ldg(&Avals[idx]) * __ldg(&v[col]);
            } else {
                srows[i] = -1;  // Invalid row marker for bounds checking
                sdata[i] = 0.0;
            }
        } else {
            srows[i] = -1;
            sdata[i] = 0.0;
        }
    }
    // OPTIMIZATION 15: Synchronize threads before processing shared memory data
    __syncthreads();
    
    // OPTIMIZATION 16: Improved warp utilization by distributing work evenly
    const int warps_per_block = blockDim.x >> 5;  // Fast division by 32
    const int items_per_warp = (slice_nnz + warps_per_block - 1) / warps_per_block;
    const int warp_start = warp_id * items_per_warp;
    const int warp_end = min(warp_start + items_per_warp, slice_nnz);
    
    // OPTIMIZATION 17: Warp-level processing for better parallelism
    // Each thread in warp processes elements with stride 32 for coalescing
    for (int i = warp_start + lane; i < warp_end; i += 32) {
        if (i >= slice_nnz) break;
        
        register int current_row = srows[i];
        if (current_row < 0 || current_row >= rows) continue;
        
        // OPTIMIZATION 18: Warp vote functions for efficient row boundary detection
        // __ballot_sync creates a bitmask of which threads satisfy the condition
        bool is_first_in_row = (i == 0) || (i > 0 && srows[i-1] != current_row);
        unsigned int first_lane_mask = __ballot_sync(0xffffffff, is_first_in_row);
        
        if (is_first_in_row) {
            register double sum = 0.0;
            
            // OPTIMIZATION 19: Sequential accumulation within same row
            // Accumulate all elements for this row efficiently in one thread
            for (int j = i; j < slice_nnz && j < warp_end && srows[j] == current_row; j++) {
                sum += sdata[j];
            }
            
            // OPTIMIZATION 20: Skip zero values to reduce atomic contention
            // Atomic operations are expensive, so avoid unnecessary ones
            if (sum != 0.0) {
                atomicAdd(&C[current_row], sum);
            }
        }
    }
}

// Simple utility functions for debugging
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

// Performance analysis function
void compute_band_gflops(int rows, int cols, int values, double time_ms, int* Acols, int estimated_slices) {
    // Calculate memory bandwidth based on actual data movement
    // SCOO format stores: row indices, column indices, values, and slice boundaries
    size_t scoo_size = (size_t)(sizeof(int) * (2 * values) + sizeof(double) * values + sizeof(int) * (estimated_slices + 1));
    
    // Count unique vector elements accessed (not all elements may be used)
    int* unique_cols = (int*)calloc(cols, sizeof(int));
    int unique_count = 0;
    for (int i=0; i<values; i++) {
        if (unique_cols[Acols[i]] == 0) {
            unique_cols[Acols[i]] = 1;
            unique_count += 1;
        }
    }
    free(unique_cols);
    
    size_t vector_size = (size_t)(sizeof(double) * unique_count);
    size_t bytes_read = scoo_size + vector_size;
    size_t bytes_written = (size_t)(sizeof(double) * rows);
    size_t total_bytes = bytes_read + bytes_written;

    // Calculate performance metrics
    double bandwidth = total_bytes / (time_ms * 1.0e6);  // GB/s
    double operations = 2.0 * values;                     // multiply + add per non-zero
    double gflops = operations / (time_ms * 1.0e6);       // GFLOPS

    printf("Bandwidth: %f GB/s\n", bandwidth);
    printf("FLOPS: %f GFLOPS\n", gflops);
}

// OPTIMIZATION 21: Comprehensive error checking macro
// Catches CUDA errors immediately for easier debugging
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// Global configuration
int ITERATIONS = 51;

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
        return 1;
    }

    FILE *fin = fopen(argv[1], "r");
    if (!fin) {
        perror("Failed to open file");
        return 1;
    }

    char buffer[100];
    int first = 1;
    double totalTime = 0.0;

    int rows, cols, values;
    int *Arows, *Acols;
    double *Avals;
    int counter = 0;

    // Variables for tracking matrix structure
    int current_row = 0;
    int max_val = 0;
    int row_counter = 0;

    // OPTIMIZATION 22: Parse Matrix Market format efficiently
    while(fgets(buffer, 100, fin)) {
        if (buffer[0] != '%') {  // Skip comment lines
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
                
                // OPTIMIZATION 23: Use cudaMallocManaged for unified memory
                // Simplifies memory management and enables automatic migration
                CUDA_CHECK(cudaMallocManaged(&Arows, values*sizeof(int)));
                CUDA_CHECK(cudaMallocManaged(&Acols, values*sizeof(int)));
                CUDA_CHECK(cudaMallocManaged(&Avals, values*sizeof(double)));
            } else {
                // Matrix Market files are 1-indexed, convert to 0-indexed
                int tmp_row = atoi(split_buffer[0])-1;
                int tmp_col = atoi(split_buffer[1])-1;
                double tmp_val = atof(split_buffer[2]);

                Arows[counter] = tmp_row;
                Acols[counter] = tmp_col;
                Avals[counter] = tmp_val;

                // Track matrix structure for optimization purposes
                if (Arows[counter] == current_row) {
                    row_counter++;
                } else if (Arows[counter] > current_row) {
                    if (row_counter > max_val) {
                        max_val = row_counter;
                    }
                    current_row = Arows[counter];
                    row_counter = 1;
                }
                counter++;
            }
        }
    }

    printf("Matrix: %d x %d with %d non-zeros\n", rows, cols, values);

    // OPTIMIZATION 24: Input validation to prevent crashes
    for (int i = 0; i < values; i++) {
        if (Arows[i] < 0 || Arows[i] >= rows || Acols[i] < 0 || Acols[i] >= cols) {
            printf("ERROR: Invalid matrix entry at index %d: (%d, %d)\n", i, Arows[i], Acols[i]);
            return 1;
        }
    }
    printf("Matrix data validation passed\n");

    // OPTIMIZATION 25: Dynamic slice sizing based on shared memory constraints
    int *Aslices;
    int num_slices = 0;
    
    // Query device properties for optimal configuration
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    int SHARED_MEM_SIZE = deviceProp.sharedMemPerBlock;
    printf("Device shared memory per block: %d bytes\n", SHARED_MEM_SIZE);
    SHARED_MEM_SIZE = (int)(SHARED_MEM_SIZE * 0.8);  // Conservative limit
    
    // OPTIMIZATION 26: Calculate optimal slice size considering alignment
    int MAX_NNZ_PER_SLICE = SHARED_MEM_SIZE / (sizeof(double) + sizeof(int) + 8);
    printf("Maximum NNZ per slice: %d\n", MAX_NNZ_PER_SLICE);

    // Allocate slice boundary array
    int estimated_slices = (values + MAX_NNZ_PER_SLICE - 1) / MAX_NNZ_PER_SLICE + 10;
    CUDA_CHECK(cudaMallocManaged(&Aslices, (estimated_slices + 1) * sizeof(int)));

    // OPTIMIZATION 27: Simple uniform slicing for predictable memory access
    // More complex slicing could preserve row boundaries but adds complexity
    Aslices[0] = 0;
    num_slices = 0;
    int max_slice_nnz = 0;

    for (int i = 0; i < values; i += MAX_NNZ_PER_SLICE) {
        Aslices[num_slices] = i;
        int slice_size = min(MAX_NNZ_PER_SLICE, values - i);
        if (slice_size > max_slice_nnz) {
            max_slice_nnz = slice_size;
        }
        num_slices++;
    }
    Aslices[num_slices] = values;

    printf("Created %d slices, max slice size: %d\n", num_slices, max_slice_nnz);

    // OPTIMIZATION 28: Shared memory sizing with safety padding
    int sharedMemSize = max_slice_nnz * (sizeof(double) + sizeof(int)) + 256;
    printf("Shared memory size: %d bytes\n", sharedMemSize);

    // Validate shared memory requirements
    if (sharedMemSize > deviceProp.sharedMemPerBlock) {
        printf("ERROR: Shared memory requirement (%d) exceeds device limit (%d)\n", 
               sharedMemSize, (int)deviceProp.sharedMemPerBlock);
        return 1;
    }

    // Create input vector (dense)
    double *v;
    CUDA_CHECK(cudaMallocManaged(&v, cols*sizeof(double)));
    for (int i=0; i<cols; i++) {
        v[i] = 1.0;  // Use 1.0 for easy result verification
    }

    // Create output vector
    double *C;
    CUDA_CHECK(cudaMallocManaged(&C, rows*sizeof(double)));
    
    // OPTIMIZATION 29: Optimal launch configuration
    int threadsPerBlock = 256;  // Multiple of warp size for full utilization
    int blocksPerGrid = num_slices;  // One block per slice

    // OPTIMIZATION 30: Check theoretical occupancy
    int maxActiveBlocks;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, spmv, threadsPerBlock, sharedMemSize));
    printf("Max active blocks per SM: %d\n", maxActiveBlocks);

    first = 1;
    cudaEvent_t start, stop;

    // OPTIMIZATION 31: Performance measurement loop
    for (int i=0; i<ITERATIONS; i++) {
        // Clear output for each iteration
        CUDA_CHECK(cudaMemset(C, 0, rows * sizeof(double)));
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        // OPTIMIZATION 32: Prefetch data to GPU for better performance
        // Unified memory benefits from explicit prefetching
        int device = -1;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaMemPrefetchAsync(Arows, values*sizeof(int), device, NULL));
        CUDA_CHECK(cudaMemPrefetchAsync(Acols, values*sizeof(int), device, NULL));
        CUDA_CHECK(cudaMemPrefetchAsync(Avals, values*sizeof(double), device, NULL));
        CUDA_CHECK(cudaMemPrefetchAsync(v, cols*sizeof(double), device, NULL));
        CUDA_CHECK(cudaMemPrefetchAsync(C, rows*sizeof(double), device, NULL));
        
        CUDA_CHECK(cudaEventRecord(start));

        // Launch optimized SCOO SpMV kernel
        spmv<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(Arows, Acols, Avals, v, C, rows, cols, values, Aslices);
        
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        float e_time = 0;
        CUDA_CHECK(cudaEventElapsedTime(&e_time, start, stop));
        
        // Skip first iteration (warmup) for accurate timing
        if (first == 1) {
            first = 0;
        } else {
            totalTime += e_time;
        }

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    // Calculate and display performance metrics
    double avg_time = totalTime / (ITERATIONS - 1);
    printf("Average time: %fms\n", avg_time);
    compute_band_gflops(rows, cols, values, avg_time, Acols, estimated_slices);

    fclose(fin);
    
    // OPTIMIZATION 33: Proper memory cleanup
    CUDA_CHECK(cudaFree(Arows));
    CUDA_CHECK(cudaFree(Acols));
    CUDA_CHECK(cudaFree(Avals));
    CUDA_CHECK(cudaFree(Aslices));
    CUDA_CHECK(cudaFree(v));
    CUDA_CHECK(cudaFree(C));

    return 0;
}