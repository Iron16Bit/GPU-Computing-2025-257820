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

// CUDA kernel - each thread computes one multiplication
__global__
void matrix_multiplication_kernel(double *A, double *B, double *C, int rows, int cols) {
    // Calculate global thread ID in a 2D grid
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if thread coordinates are within matrix bounds
    if (row < rows && col < cols) {
        // Compute one multiplication and add atomically to result
        double product = A[row * cols + col] * B[col];
        if (product != 0.0) { // Only add non-zero contributions
            atomicAdd(&C[row], product);
        }
    }
}

double calculateBandwidthGBs(int rows, int cols, double timeMs) {
    double matrix_size = rows * cols * sizeof(double); // Dense matrix size
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
                
                // Use cudaMallocManaged for unified memory
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

    // Create dense vector using unified memory
    double *v;
    cudaMallocManaged(&v, cols*sizeof(double));
    for (int i=0; i<cols; i++) {
        v[i] = 1.0;
    }
    
    // Naive solution: create a sparse matrix from the COO using unified memory
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
    
    // Perform GPU matrix-vector multiplication
    // Set up 2D grid to match matrix dimensions
    dim3 threadsPerBlock(16, 16);  // 16x16 = 256 threads per block
    dim3 numBlocks(
        (cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (rows + threadsPerBlock.y - 1) / threadsPerBlock.y
    );
    
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
    matrix_multiplication_kernel<<<numBlocks, threadsPerBlock>>>(M, v, C, rows, cols);
    cudaEventRecord(stop);
    
    // Wait for kernel to complete
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("[GPU naive] Elapsed time: %f ms\n", milliseconds);
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