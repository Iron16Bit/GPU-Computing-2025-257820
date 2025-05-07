#include <stdlib.h>
#include <stdio.h>
#include <string.h>

__global__
void spmv(int *Arows, int *Acols, double *Avals, double *v, double *C, int rows, int cols, int values) {
    int row = blockIdx.x;
    int lane = threadIdx.x % 32;
    
    if (row < rows) {
        double sum = 0.0;
        
        // Binary search to find the starting position for this row
        int left = 0;
        int right = values - 1;
        int start_pos = 0;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (Arows[mid] < row) {
                left = mid + 1;
            } else if (Arows[mid] > row) {
                right = mid - 1;
            } else {
                // We found a match, but need to find the first occurrence
                start_pos = mid;
                right = mid - 1;
            }
        }
        
        // If we didn't find an exact match, left is the insertion point
        if (Arows[start_pos] != row && left < values) {
            start_pos = left;
        }
        
        // Each thread processes elements in stride
        for (int i = start_pos + lane; i < values && Arows[i] == row; i += 32) {
            sum += Avals[i] * __ldg(&v[Acols[i]]);
        }
        
        // Warp-level reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        if (lane == 0) {
            C[row] = sum;
        }
    }
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

int ITERATIONS = 11;

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
                
                // Use cudaMallocManaged instead of malloc
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
    
    // Perform SpMV
    int threadsPerBlock = 32;  // One warp per row
    int blocksPerGrid = rows;  // One block per row

    cudaEvent_t start, stop;

    for (int i=0; i<ITERATIONS; i++) {
        cudaMemset(C, 0, rows * sizeof(double));
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        // Prefetch data to GPU
        int device = -1;
        cudaGetDevice(&device);
        cudaMemPrefetchAsync(Arows, values*sizeof(int), device, NULL);
        cudaMemPrefetchAsync(Acols, values*sizeof(int), device, NULL);
        cudaMemPrefetchAsync(Avals, values*sizeof(double), device, NULL);
        cudaMemPrefetchAsync(v, cols*sizeof(double), device, NULL);
        cudaMemPrefetchAsync(C, rows*sizeof(double), device, NULL);
        
        cudaEventRecord(start);

        spmv<<<blocksPerGrid, threadsPerBlock>>>(Arows, Acols, Avals, v, C, rows, cols, values);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        // Ensure all operations are completed
        cudaDeviceSynchronize();
        
        float e_time = 0;
        cudaEventElapsedTime(&e_time, start, stop);
        // print_double_array(C, rows);
        printf("Kernel completed in %fms\n", e_time);
        if (first == 1) {
            first = 0;
        } else {
            totalTime += e_time;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    // print_double_array(C, rows);

    // Calculate average time
    double avg_time = totalTime / (ITERATIONS - 1);
    printf("Average time: %fms\n", avg_time);

    fclose(fin);
    
    // Free using cudaFree instead of free
    cudaFree(Arows);
    cudaFree(Acols);
    cudaFree(Avals);
    cudaFree(v);
    cudaFree(C);

    return 0;
}