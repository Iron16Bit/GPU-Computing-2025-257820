#include <stdlib.h>
#include <stdio.h>
#include <string.h>

__global__
void spmv(int *Arows, int *Acols, double *Avals, double *v, double *C, int rows, int cols, int values, int *colOffsets) {
    int current_col = blockIdx.x * blockDim.x + threadIdx.x;
    int t = v[current_col];
    
    if (current_col < cols) {
        int start = colOffsets[current_col];
        int end = (current_col < cols - 1) ? colOffsets[current_col + 1] : values;

        for (int i = start; i < end; i++) {
            // All elements in this range belong to current_col, no need to check
            double product = Avals[i] * v[current_col];
            atomicAdd(&C[Arows[i]], product);
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

void swap(int* Arows, int* Acols, double* Avals, int i, int j) {
    int tmp_row = Arows[i];
    int tmp_col = Acols[i];
    double tmp_val = Avals[i];

    Arows[i] = Arows[j];
    Acols[i] = Acols[j];
    Avals[i] = Avals[j];

    Arows[j] = tmp_row;
    Acols[j] = tmp_col;
    Avals[j] = tmp_val;
}

void sort(int* Arows, int* Acols, double* Avals, int n) {
    for (int i=0; i<n-1; i++) {
        for (int j=i+1; j<n; j++) {
            if (Acols[i] > Acols[j]) {
                swap(Arows, Acols, Avals, i, j);
            } else if ((Acols[i] == Acols[j]) && (Arows[i] > Arows[j])) {
                swap(Arows, Acols, Avals, i, j);
            }
        }
    }
}

double calculateBandwidthGBs(int values, int rows, double timeMs) {
    double COO_size = values * (sizeof(int) + sizeof(int) + sizeof(double)); // COO size in bytes
    double vector_size = rows * sizeof(double); // Dense vector size in bytes
    double bytesAccessed = COO_size + vector_size;

    // Convert ms to seconds and bytes to GB
    double timeS = timeMs * 1e-3;
    double dataGB = bytesAccessed * 1e-9;
    
    return dataGB / timeS;
}

int ITERATIONS = 10;

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

    // Sort COO
    sort(Arows, Acols, Avals, values);

    // After sorting, create column offsets array
    int *colOffsets;
    cudaMallocManaged(&colOffsets, (cols + 1) * sizeof(int));
    
    // Initialize all offsets to -1 (meaning no elements for that column)
    for (int i = 0; i < cols + 1; i++) {
        colOffsets[i] = -1;
    }
    
    // Set the offset for each column to the first occurrence of that column
    for (int i = 0; i < values; i++) {
        if (colOffsets[Acols[i]] == -1) {
            colOffsets[Acols[i]] = i;
        }
    }
    
    // Fill in any columns that don't have elements with the next valid offset
    int lastValidOffset = values;
    for (int i = cols - 1; i >= 0; i--) {
        if (colOffsets[i] == -1) {
            colOffsets[i] = lastValidOffset;
        } else {
            lastValidOffset = colOffsets[i];
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
    int N = cols;
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;

    for (int i=0; i<ITERATIONS; i++) {
        cudaMemset(C, 0, rows * sizeof(double));
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        // Prefetch data to GPU (including the column offsets)
        int device = -1;
        cudaGetDevice(&device);
        cudaMemPrefetchAsync(Arows, values*sizeof(int), device, NULL);
        cudaMemPrefetchAsync(Acols, values*sizeof(int), device, NULL);
        cudaMemPrefetchAsync(Avals, values*sizeof(double), device, NULL);
        cudaMemPrefetchAsync(v, cols*sizeof(double), device, NULL);
        cudaMemPrefetchAsync(C, rows*sizeof(double), device, NULL);
        cudaMemPrefetchAsync(colOffsets, (cols+1)*sizeof(int), device, NULL);
        
        cudaEventRecord(start);

        spmv<<<blocksPerGrid, threadsPerBlock>>>(Arows, Acols, Avals, v, C, rows, cols, values, colOffsets);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        // Ensure all operations are completed
        cudaDeviceSynchronize();
        
        float e_time = 0;
        cudaEventElapsedTime(&e_time, start, stop);
        printf("Kernel completed in %fms\n", e_time);
        totalTime += e_time;
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    // Calculate average time
    double avg_time = totalTime / ITERATIONS;
    printf("Average time: %fms\n", avg_time);
    printf("Bandwidth: %f GB/s\n", calculateBandwidthGBs(values, rows, avg_time));

    fclose(fin);
    
    // Free using cudaFree (don't forget the column offsets)
    cudaFree(Arows);
    cudaFree(Acols);
    cudaFree(Avals);
    cudaFree(v);
    cudaFree(C);
    cudaFree(colOffsets);

    return 0;
}