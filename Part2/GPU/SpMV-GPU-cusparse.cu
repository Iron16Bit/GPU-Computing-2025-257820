#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cusparse.h>

// Reference basic SpMV kernel for comparison
__global__
void spmv(int *Arows, int *Acols, double *Avals, double *v, double *C, int rows, int cols, int values) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < values) {
        double product = Avals[tid] * v[Acols[tid]];
        atomicAdd(&C[Arows[tid]], product);
    }
}

// Compute bandwidth and flops
void compute_band_gflops(int rows, int cols, int values, double time_ms, int* Acols) {
    // Bytes read from the COO
    size_t coo_size = (size_t)(sizeof(int) * (2 * values) + sizeof(double) * values);
    // Bytes read from the dense vector
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
    // Total bytes read
    size_t bytes_read = coo_size + vector_size;
    // Bytes written
    size_t bytes_written = (size_t)(sizeof(double) * rows);
    size_t total_bytes = bytes_read + bytes_written;

    // GFLOPS
    double bandwidth = total_bytes / (time_ms * 1.0e6);
    double operations = 2.0 * values;
    double gflops = operations / (time_ms * 1.0e6);

    printf("Bandwidth: %f GB/s\n", bandwidth);
    printf("FLOPS: %f GFLOPS\n", gflops);
}

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

#define ITERATIONS 51
#define DEFAULT_THREADS_PER_BLOCK 256

int main(int argc, char *argv[]) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "Usage: %s <input_file> [threads_per_block]\n", argv[0]);
        return 1;
    }

    printf("Used matrix: %s\n", argv[1]);

    FILE *fin = fopen(argv[1], "r");

    if (!fin) {
        perror("Failed to open file");
        return 1;
    }

    char buffer[100];
    int first = 1;
    double totalTime = 0.0;

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
                
                // Use cudaMallocManaged for COO data
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

    // Create dense vector using cudaMallocManaged
    double *v;
    cudaMallocManaged(&v, cols*sizeof(double));
    for (int i=0; i<cols; i++) {
        v[i] = 1.0;
    }

    // Create output vector using cudaMallocManaged
    double *C;
    cudaMallocManaged(&C, rows*sizeof(double));
    
    // Initialize cuSPARSE
    cusparseHandle_t handle;
    cusparseCreate(&handle);
    
    // Create matrix descriptor for COO format
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    
    // Create sparse matrix in COO format
    cusparseCreateCoo(&matA, rows, cols, values,
                      Arows, Acols, Avals,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    
    // Create dense vectors
    cusparseCreateDnVec(&vecX, cols, v, CUDA_R_64F);
    cusparseCreateDnVec(&vecY, rows, C, CUDA_R_64F);
    
    // Scalars for SpMV operation: y = alpha * A * x + beta * y
    double alpha = 1.0, beta = 0.0;
    
    // Calculate buffer size for SpMV
    size_t bufferSize = 0;
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                           &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                           CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
    
    // Allocate buffer
    void* dBuffer = NULL;
    if (bufferSize > 0) {
        cudaMalloc(&dBuffer, bufferSize);
    }

    cudaEvent_t start, stop;
    first = 1;

    for (int i=0; i<ITERATIONS; i++) {
        // Reset output vector
        cudaMemset(C, 0, rows * sizeof(double));
        
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);

        // Perform SpMV using cuSPARSE: C = alpha * A * v + beta * C
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                     CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float e_time = 0;
        cudaEventElapsedTime(&e_time, start, stop);
        
        if (first == 1) {
            first = 0;
        } else {
            totalTime += e_time;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Calculate average time
    double avg_time = totalTime / (ITERATIONS - 1);
    printf("Average time: %fms\n", avg_time);
    compute_band_gflops(rows, cols, values, avg_time, Acols);

    // Cleanup cuSPARSE resources
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(handle);
    
    if (dBuffer) {
        cudaFree(dBuffer);
    }

    fclose(fin);
    
    // Free memory
    cudaFree(Arows);
    cudaFree(Acols);
    cudaFree(Avals);
    cudaFree(v);
    cudaFree(C);

    return 0;
}