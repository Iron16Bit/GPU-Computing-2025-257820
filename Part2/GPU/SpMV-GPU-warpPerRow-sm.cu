#include <stdlib.h>
#include <stdio.h>
#include <string.h>

__global__
void spmv(int *row_pointer, int *Acols, double *Avals, double *v, double *C, int rows, int cols, int values) {
    int row = blockIdx.x;
    int lane = threadIdx.x;
    
    // Shared memory for row data
    extern __shared__ char shared_mem[];
    int *shared_cols = (int*)shared_mem;
    double *shared_vals = (double*)(shared_mem + 32 * sizeof(int));
    
    if (row < rows && threadIdx.x < 32) {
        double sum = 0.0;
        int start_pos = row_pointer[row];
        int end_pos = row_pointer[row+1];
        int row_nnz = end_pos - start_pos;
        
        // Load row data into shared memory in chunks
        for (int chunk_start = 0; chunk_start < row_nnz; chunk_start += 32) {
            int chunk_size = min(32, row_nnz - chunk_start);
            
            // Cooperatively load chunk into shared memory
            if (lane < chunk_size) {
                int global_idx = start_pos + chunk_start + lane;
                shared_cols[lane] = Acols[global_idx];
                shared_vals[lane] = Avals[global_idx];
            }
            __syncwarp();
            
            // Process chunk from shared memory
            if (lane < chunk_size) {
                sum += shared_vals[lane] * __ldg(&v[shared_cols[lane]]);
            }
            __syncwarp();
        }
        
        // Warp-level reduction with loop unrolling
        sum += __shfl_down_sync(0xffffffff, sum, 16);
        sum += __shfl_down_sync(0xffffffff, sum, 8);
        sum += __shfl_down_sync(0xffffffff, sum, 4);
        sum += __shfl_down_sync(0xffffffff, sum, 2);
        sum += __shfl_down_sync(0xffffffff, sum, 1);
        
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
        printf("%f\n", a[i]);
    }
}

void print_matrix(double* m, int rows, int cols) {
    for (int i=0; i<rows*cols; i++) {
        printf("%f\n", m[i]);
    }
}

// Compute bandwidth and flops
void compute_band_gflops(int rows, int cols, int values, double time_ms, int* Acols) {
    // Bytes read from the CSR
    size_t csr_size = (size_t)(sizeof(int) * values + sizeof(int) * (rows+1) + sizeof(double) * values);
    // Bytes read from the dense vector
    int* unique_cols = (int*)calloc(cols, sizeof(int));
    int unique_count = 0;
    for (int i=0; i<values; i++) {
        if (unique_cols[Acols[i]] == 0) {
            unique_cols[Acols[i]] = 1;
            unique_count += 1;
        }
    }
    size_t vector_size = (size_t)(sizeof(double) * unique_count);
    // Total bytes read
    size_t bytes_read = csr_size + vector_size;
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


void convert_coo_to_csr(int *Arows, int *Acols, double *Avals, 
                        int **csr_cols, double **csr_vals, int *row_pointer, 
                        int values, int rows) {
    // Initialize row pointer
    for (int i = 0; i <= rows; i++) {
        row_pointer[i] = 0;
    }
    
    // Count elements per row with bounds checking
    for (int i = 0; i < values; i++) {
        if (Arows[i] >= 0 && Arows[i] < rows) {
            row_pointer[Arows[i] + 1]++;
        }
    }
    
    // Convert counts to cumulative offsets
    for (int i = 1; i <= rows; i++) {
        row_pointer[i] += row_pointer[i-1];
    }
    
    // Allocate CSR arrays
    cudaMallocManaged(csr_cols, values * sizeof(int));
    cudaMallocManaged(csr_vals, values * sizeof(double));
    
    // Create temporary copy of row_pointer for insertion
    int *temp_row_ptr = (int*)malloc((rows + 1) * sizeof(int));
    for (int i = 0; i <= rows; i++) {
        temp_row_ptr[i] = row_pointer[i];
    }
    
    // Fill CSR arrays with bounds checking
    for (int i = 0; i < values; i++) {
        int row = Arows[i];
        if (row >= 0 && row < rows) {
            int pos = temp_row_ptr[row]++;
            if (pos < values) {
                (*csr_cols)[pos] = Acols[i];
                (*csr_vals)[pos] = Avals[i];
            }
        }
    }
    
    free(temp_row_ptr);
}

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

    // Convert COO to CSR
    int* row_pointer;
    int* csr_cols;
    double* csr_vals;
    cudaMallocManaged(&row_pointer, (rows + 1) * sizeof(int));
    
    convert_coo_to_csr(Arows, Acols, Avals, &csr_cols, &csr_vals, row_pointer, values, rows);
    
    // Free original COO arrays
    cudaFree(Arows);
    cudaFree(Acols);
    cudaFree(Avals);

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
        
        cudaEventRecord(start);

        int shared_mem_size = 32 * (sizeof(int) + sizeof(double)); // For approach 2
        spmv<<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>(row_pointer, csr_cols, csr_vals, v, C, rows, cols, values);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        // Ensure all operations are completed
        cudaDeviceSynchronize();
        
        float e_time = 0;
        cudaEventElapsedTime(&e_time, start, stop);
        // print_double_array(C, rows);
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
    compute_band_gflops(rows, cols, values, avg_time, csr_cols);

    fclose(fin);
    
    // Free using cudaFree instead of free
    cudaFree(csr_cols);
    cudaFree(csr_vals);
    cudaFree(row_pointer);
    cudaFree(v);
    cudaFree(C);

    return 0;
}