#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// Adjusted constant for your GPU (which has 48KB shared memory)
#define OPTIMAL_SHARED_MEM_USAGE 49152  // Use ~48 KB (maximum for your GPU)

__global__
void spmv(int *Arows, int *Acols, double *Avals, double *v, double *C, int rows, int cols, int values, int *Aslices) {
    // Compute slice info
    int slice_id = blockIdx.x;
    int tid = threadIdx.x;
    int slice_start = Aslices[slice_id];
    int slice_end = Aslices[slice_id + 1];
    int slice_nnz = slice_end - slice_start;

    // Shared memory layout - only using two arrays
    extern __shared__ char shared[];
    double* sdata = (double*)shared;
    int* srows = (int*)&sdata[slice_nnz];

    // Fill shared memory
    for (int i = tid; i < slice_nnz; i += blockDim.x) {
        int idx = slice_start + i;
        srows[i] = Arows[idx];
        sdata[i] = Avals[idx] * v[Acols[idx]]; // Direct access to global v
    }
    __syncthreads();

    // Reduction phase: accumulate results by row
    for (int i = tid; i < slice_nnz; i += blockDim.x) {
        // Only the last occurrence of a row in the slice writes the sum
        if (i == slice_nnz - 1 || srows[i] != srows[i + 1]) {
            double sum = sdata[i];
            int row = srows[i];
            int j = i - 1;
            // Accumulate backwards for all previous elements with the same row
            while (j >= 0 && srows[j] == row) {
                sum += sdata[j];
                j--;
            }
            atomicAdd(&C[row], sum);
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
            if (Arows[i] > Arows[j]) {
                swap(Arows, Acols, Avals, i, j);
            } else if ((Arows[i] == Arows[j]) && (Acols[i] > Acols[j])) {
                swap(Arows, Acols, Avals, i, j);
            }
        }
    }
}

double calculateBandwidthGBs(int values, int rows, int cols, double timeMs) {
    double COO_size = values * (sizeof(int) + sizeof(int) + sizeof(double)); // COO size in bytes
    double vector_size = cols * sizeof(double); // Dense vector size in bytes
    double output_size = rows * sizeof(double); // Output vector size in bytes
    double bytesAccessed = COO_size + vector_size + output_size;

    // Convert ms to seconds and bytes to GB
    double timeS = timeMs * 1e-3;
    double dataGB = bytesAccessed * 1e-9;
    
    return dataGB / timeS;
}

// GLOBAL VARIABLES
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

    // Sort COO
    sort(Arows, Acols, Avals, values);

    // Build SCOO format adjusted for actual GPU shared memory
    int *Aslices;
    int num_slices = 0;
    
    // Get device properties first to determine shared memory limit
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int SHARED_MEM_SIZE = deviceProp.sharedMemPerBlock;
    
    printf("Device shared memory per block: %d bytes\n", SHARED_MEM_SIZE);
    
    // Use slightly less than the maximum (leave room for register spilling)
    SHARED_MEM_SIZE = (int)(SHARED_MEM_SIZE * 0.95);
    
    // Calculate maximum elements per slice that will fit in shared memory
    int MAX_NNZ_PER_SLICE = SHARED_MEM_SIZE / (sizeof(double) + sizeof(int));
    printf("Maximum NNZ per slice: %d\n", MAX_NNZ_PER_SLICE);
    
    // Allocate worst-case space for Aslices
    cudaMallocManaged(&Aslices, (values + 1) * sizeof(int));

    int nnz_in_slice = 0;
    int max_slice_nnz = 0;
    Aslices[0] = 0;
    num_slices = 1;

    // Create slices that fit in shared memory
    for (int i = 0; i < values; ++i) {
        nnz_in_slice++;
        bool next_is_new_row = (i < values - 1 && Arows[i] != Arows[i + 1]);

        // Create a new slice if:
        // 1. Current slice is near capacity AND we're at a row boundary
        // 2. Or we're absolutely at capacity
        if ((nnz_in_slice >= MAX_NNZ_PER_SLICE * 0.9 && next_is_new_row) || 
            (nnz_in_slice >= MAX_NNZ_PER_SLICE)) {
            Aslices[num_slices] = i + 1;
            if (nnz_in_slice > max_slice_nnz) max_slice_nnz = nnz_in_slice;
            nnz_in_slice = 0;
            num_slices++;
        }
    }

    // Ensure the last slice ends at the last element
    if (Aslices[num_slices - 1] != values) {
        Aslices[num_slices] = values;
        if (nnz_in_slice > max_slice_nnz) max_slice_nnz = nnz_in_slice;
        num_slices++;
    }

    // Calculate shared memory size 
    int sharedMemSize = max_slice_nnz * sizeof(double)  // For sdata
                      + max_slice_nnz * sizeof(int);    // For srows
    
    printf("Sliced matrix into %d slices, max slice size: %d NNZ\n", 
           num_slices, max_slice_nnz);
    printf("Using shared memory size: %d bytes per block\n", sharedMemSize);

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
    // Use a warp per slice
    int threadsPerBlock = 256;  // A30 supports up to 1024, but 256 is often optimal
    int blocksPerGrid = num_slices;

    first = 1;

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

        spmv<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(Arows, Acols, Avals, v, C, rows, cols, values, Aslices);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
        }
        
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
    double avg_time = totalTime / ITERATIONS;
    printf("Average time: %fms\n", avg_time);
    printf("Bandwidth: %f GB/s\n", calculateBandwidthGBs(values, rows, cols, avg_time));

    fclose(fin);
    
    // Free using cudaFree instead of free
    cudaFree(Arows);
    cudaFree(Acols);
    cudaFree(Avals);
    cudaFree(v);
    cudaFree(C);

    return 0;
}