#include <stdlib.h>
#include <stdio.h>
#include <string.h>

__global__
void spmv(int *Arows, int *Acols, double *Avals, double *v, double *C, int rows, int cols, int values) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 
    
    if (tid < values) {
        int total_threads = gridDim.x * blockDim.x;
        
        int vals_per_thread = (values + total_threads - 1) / total_threads; 
        int start = tid * vals_per_thread;                                 
        int end = min(start + vals_per_thread, values);                    

        if (start < values) {
            for (int i = start; i < end; i++) {
                if (i < values) { 
                    double product = Avals[i] * v[Acols[i]]; 
                    atomicAdd(&C[Arows[i]], product);
                }
            }
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

// double calculateBandwidthGBs(int values, int rows, int cols, double timeMs) {
//     double COO_size = values * (sizeof(int) + sizeof(int) + sizeof(double)); // COO size in bytes
//     double vector_size = cols * sizeof(double); // Dense vector size in bytes
//     double output_size = rows * sizeof(double); // Output vector size in bytes
//     double bytesAccessed = COO_size + vector_size + output_size;

//     // Convert ms to seconds and bytes to GB
//     double timeS = timeMs * 1e-3;
//     double dataGB = bytesAccessed * 1e-9;
    
//     return dataGB / timeS;
// }

#define ITERATIONS 51
#define DEFAULT_THREADS 256
#define DEFAULT_BLOCKS 4

int main(int argc, char *argv[]) {
    if (argc < 2 || argc > 4) {
        fprintf(stderr, "Usage: %s <input_file> [num_threads] [num_blocks]\n", argv[0]);
        return 1;
    }

    int THREADS = DEFAULT_THREADS;
    int BLOCKS = DEFAULT_BLOCKS;

    // Parse threads parameter
    if (argc >= 3) {
        int user_threads = atoi(argv[2]);
        if (user_threads > 0) {
            THREADS = user_threads;
        } else {
            fprintf(stderr, "Warning: Invalid number of threads, using default (%d)\n", DEFAULT_THREADS);
        }
    }

    // Parse blocks parameter
    if (argc >= 4) {
        int user_blocks = atoi(argv[3]);
        if (user_blocks > 0) {
            BLOCKS = user_blocks;
        } else {
            fprintf(stderr, "Warning: Invalid number of blocks, using default (%d)\n", DEFAULT_BLOCKS);
        }
    }

    printf("Using configuration: %d threads per block, %d blocks\n", THREADS, BLOCKS);

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

    cudaEvent_t start, stop;

    first = 1;

    for (int i=0; i<ITERATIONS; i++) {
        cudaMemset(C, 0, rows * sizeof(double));
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);

        spmv<<<BLOCKS, THREADS>>>(Arows, Acols, Avals, v, C, rows, cols, values);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
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
    // printf("Bandwidth: %f GB/s\n", calculateBandwidthGBs(values, rows, cols, avg_time));

    fclose(fin);
    
    // Free using cudaFree instead of free
    cudaFree(Arows);
    cudaFree(Acols);
    cudaFree(Avals);
    cudaFree(v);
    cudaFree(C);

    return 0;
}