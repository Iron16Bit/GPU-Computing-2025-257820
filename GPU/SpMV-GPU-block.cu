#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

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

// CUDA kernel for block-based SpMV
__global__
void block_spmv_kernel(double *M, double *v, double *C, int rows, int cols, 
                        int row_block, int col_block, int blocks_x) {
    // Calculate which block this thread block is processing
    int block_id = blockIdx.y * gridDim.x + blockIdx.x;
    
    // Calculate offsets based on block ID
    int c_offset = block_id / blocks_x * row_block;
    int v_offset = block_id % blocks_x * col_block;
    
    // Get thread ID within the block
    int     ;
    int num_threads = blockDim.x * blockDim.y;
    
    // Shared memory for the vector chunk and partial results
    extern __shared__ double shared_mem[];
    double* shared_v = shared_mem;
    double* block_results = &shared_mem[col_block];
    
    // Initialize block results
    if (tid < row_block) {
        block_results[tid] = 0.0;
    }
    
    // Load vector chunk into shared memory
    for (int i = tid; i < col_block; i += num_threads) {
        if (v_offset + i < cols) {
            shared_v[i] = v[v_offset + i];
        }
    }
    __syncthreads();
    
    // Compute matrix-vector product for this block
    // Each thread handles multiple rows if needed
    for (int i = tid; i < row_block; i += num_threads) {
        if (c_offset + i < rows) {
            double sum = 0.0;
            for (int k = 0; k < col_block; k++) {
                if (v_offset + k < cols) {
                    int m_idx = (c_offset + i) * cols + (v_offset + k);
                    sum += M[m_idx] * shared_v[k];
                }
            }
            // Store in shared memory
            block_results[i] = sum;
        }
    }
    __syncthreads();
    
    // Accumulate results to global memory
    for (int i = tid; i < row_block; i += num_threads) {
        if (c_offset + i < rows && block_results[i] != 0.0) {
            atomicAdd(&C[c_offset + i], block_results[i]);
        }
    }
}

// Smart determination of optimal block sizes
void determine_optimal_blocks(int rows, int cols, int* row_block, int* col_block) {
    // First check if we can divide evenly
    int max_row_divisor = 1;
    int max_col_divisor = 1;
    
    // Find divisors that are powers of 2 up to 256 (common block size limit)
    for (int i = 1; i <= 256; i *= 2) {
        if (rows % i == 0 && i <= rows) max_row_divisor = i;
        if (cols % i == 0 && i <= cols) max_col_divisor = i;
    }
    
    // If we can't divide evenly, find largest divisor that's a power of 2
    if (max_row_divisor == 1) {
        for (int i = 256; i >= 1; i /= 2) {
            if (rows >= i) {
                max_row_divisor = i;
                break;
            }
        }
    }
    
    if (max_col_divisor == 1) {
        for (int i = 256; i >= 1; i /= 2) {
            if (cols >= i) {
                max_col_divisor = i;
                break;
            }
        }
    }
    
    // Consider GPU characteristics (usually 32 threads per warp)
    // Let's try to make block sizes multiples of 32
    if (max_row_divisor >= 32) {
        *row_block = max_row_divisor;
    } else {
        *row_block = max_row_divisor * (32 / max_row_divisor + (32 % max_row_divisor == 0 ? 0 : 1));
        *row_block = (*row_block > rows) ? rows : *row_block;
    }
    
    if (max_col_divisor >= 32) {
        *col_block = max_col_divisor;
    } else {
        *col_block = max_col_divisor * (32 / max_col_divisor + (32 % max_col_divisor == 0 ? 0 : 1));
        *col_block = (*col_block > cols) ? cols : *col_block;
    }
}

// Calculate bandwidth in GB/s
double calculateBandwidthGBs(int rows, int cols, double timeMs) {
    double matrix_size = rows * cols * sizeof(double); // Matrix size
    double vector_size = cols * sizeof(double); // Vector size
    double output_size = rows * sizeof(double); // Output vector size
    double bytesAccessed = matrix_size + vector_size + output_size;

    // Convert ms to seconds and bytes to GB
    double timeS = timeMs * 1e-3;
    double dataGB = bytesAccessed * 1e-9;
    
    return dataGB / timeS;
}

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
                
                // Use CUDA managed memory
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

    // Create dense vector
    double *v;
    cudaMallocManaged(&v, cols*sizeof(double));
    for (int i=0; i<cols; i++) {
        v[i] = 1.0;
    }
    
    // Create matrix and result vector
    double *M;
    double *C;
    cudaMallocManaged(&M, rows*cols*sizeof(double));
    cudaMallocManaged(&C, rows*sizeof(double));
    
    // Initialize memory
    cudaMemset(M, 0, rows*cols*sizeof(double));
    cudaMemset(C, 0, rows*sizeof(double));
    
    // Convert COO to dense format
    for(int i=0; i<values; i++) {
        M[Arows[i]*cols+Acols[i]] = Avals[i];
    }
    
    // Smart determination of block sizes
    int row_block, col_block;
    determine_optimal_blocks(rows, cols, &row_block, &col_block);
    printf("Using row_block = %d, col_block = %d\n", row_block, col_block);
    
    // Calculate grid dimensions
    int blocks_x = (cols + col_block - 1) / col_block;
    int blocks_y = (rows + row_block - 1) / row_block;
    int total_blocks = blocks_x * blocks_y;
    
    // Configure thread block dimensions (matching block processing size where possible)
    int thread_block_x = min(32, col_block); // Use 32 as max for good warp alignment
    int thread_block_y = min(8, row_block);  // 8 rows per thread block gives 256 threads
    dim3 threadsPerBlock(thread_block_x, thread_block_y);
    
    // Configure grid dimensions to process all blocks
    dim3 blocksPerGrid(min(blocks_x, 65535), min(blocks_y, 65535));
    
    // Shared memory size
    int sharedMemSize = col_block * sizeof(double) + row_block * sizeof(double);
    
    // Setup timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Prefetch data to GPU
    int device = -1;
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(M, rows*cols*sizeof(double), device, NULL);
    cudaMemPrefetchAsync(v, cols*sizeof(double), device, NULL);
    cudaMemPrefetchAsync(C, rows*sizeof(double), device, NULL);
    
    // Run kernel with timing
    cudaEventRecord(start);
    block_spmv_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
        M, v, C, rows, cols, row_block, col_block, blocks_x);
    cudaEventRecord(stop);
    
    // Wait for kernel to complete
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("[GPU block] Elapsed time: %f ms\n", milliseconds);
    printf("Bandwidth: %f GB/s\n", calculateBandwidthGBs(rows, cols, milliseconds));
    
    // Cleanup events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    print_double_array(C, rows);
    
    fclose(fin);

    // Free using cudaFree
    cudaFree(Arows);
    cudaFree(Acols);
    cudaFree(Avals);
    cudaFree(v);
    cudaFree(M);
    cudaFree(C);

    return 0;
}