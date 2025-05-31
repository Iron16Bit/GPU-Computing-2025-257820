#include <stdlib.h>
#include <stdio.h>
#include <string.h>

__global__
void spmv(int *Arows, int *Acols, double *Avals, double *v, double *C, int rows, int cols, int values, int *Aslices) {
    int slice_id = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;
    
    int slice_start = Aslices[slice_id];
    int slice_end = Aslices[slice_id + 1];
    int slice_nnz = slice_end - slice_start;

    // Shared memory layout for this slice
    // First half for values, second half for row indices
    extern __shared__ char shared[];
    double* sdata = (double*)shared;
    int* srows = (int*)&sdata[slice_nnz];

    for (int i = tid; i < slice_nnz; i += blockDim.x) {
        int idx = slice_start + i;
        srows[i] = Arows[idx];
        sdata[i] = Avals[idx] * __ldg(&v[Acols[idx]]);
    }
    __syncthreads();
    
    if (slice_nnz > 0) {
        // Each warp handles a section of the slice
        int items_per_warp = (slice_nnz + (blockDim.x/32) - 1) / (blockDim.x/32);
        int warp_start = warp_id * items_per_warp;
        int warp_end = min(warp_start + items_per_warp, slice_nnz);
        
        // Process only unique rows in our section
        for (int i = warp_start; i < warp_end; i++) {
            // Check if this is a new row or the first element
            bool is_new_row = (i == 0) || (srows[i] != srows[i-1]);
            
            if (is_new_row) {
                int row = srows[i];
                int row_start = i;
                
                // Find end of this row's section
                int row_end = row_start + 1;
                while (row_end < slice_nnz && srows[row_end] == row) {
                    row_end++;
                }
                
                // Each thread in warp processes elements in stride
                double sum = 0.0;
                for (int j = row_start + lane; j < row_end; j += 32) {
                    sum += sdata[j];
                }
                
                // Warp-level reduction
                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) {
                    sum += __shfl_down_sync(0xffffffff, sum, offset);
                }
                
                // First thread writes result
                if (lane == 0) {
                    atomicAdd(&C[row], sum);
                }
                
                // Skip to next row
                i = row_end - 1;
            }
        }
    }
}

// Compute bandwidth and flops
void compute_band_gflops(int rows, int cols, int values, double time_ms, int* Acols, int estimated_slices) {
    // Bytes read from the CSR
    size_t scoo = (size_t)(sizeof(int) * (2*values) + sizeof(int) * (estimated_slices+1) + sizeof(double) * values);
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
    size_t bytes_read = scoo + vector_size;
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

// GLOBAL VARIABLES
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

    // Variables for SCOO format. Get longest row
    int current_row = 0;
    int max_row;
    int max_val = 0;
    int row_counter = 0;

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

                if (Arows[counter] == current_row) {
                    row_counter++;
                } else if (Arows[counter] > current_row) {
                    if (row_counter > max_val) {
                        max_val = row_counter;
                        max_row = current_row;
                    }
                    current_row = Arows[counter];
                    row_counter = 1;
                }

                counter+=1;
            }
        }
    }

    // Build SCOO
    int *Aslices;
    int num_slices = 0;
    
    // Get device shared memory size. We want to make slices that use most of it
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int SHARED_MEM_SIZE = deviceProp.sharedMemPerBlock;
    printf("Device shared memory per block: %d bytes\n", SHARED_MEM_SIZE);
    // Use slightly less than the maximum (leave room for register spilling)
    SHARED_MEM_SIZE = (int)(SHARED_MEM_SIZE * 0.95);
    
    // Calculate maximum elements per slice that will fit in shared memory
    int MAX_NNZ_PER_SLICE = SHARED_MEM_SIZE / (sizeof(double) + sizeof(int));
    printf("Maximum NNZ per slice: %d\n", MAX_NNZ_PER_SLICE);

    // Estimate how many slices we need
    // (Slices needed to fit all values) + (Slices needed for rows longer than a slice)
    int estimated_slices = ((values + MAX_NNZ_PER_SLICE - 1) / MAX_NNZ_PER_SLICE )+ (max_val / MAX_NNZ_PER_SLICE + 1);
    cudaMallocManaged(&Aslices, (estimated_slices + 1) * sizeof(int));

    // Initialize slicing
    Aslices[0] = 0;
    num_slices = 1;
    int nnz_in_slice = 0;
    int max_slice_nnz = 0;
    int last_row = -1;
    int row_start_idx = 0;
    int elements_in_current_row = 0;

    for (int i = 0; i < values; i++) {
        int current_row = Arows[i];
        
        // If we've moved to a new row
        if (current_row != last_row) {
            // If we were tracking a row, update its count
            if (last_row != -1) {
                elements_in_current_row = i - row_start_idx;
                
                // If this row is too long for a single slice, we need to split it
                if (elements_in_current_row > MAX_NNZ_PER_SLICE) {
                    // Create slices for this long row
                    int remaining = elements_in_current_row;
                    int pos = row_start_idx;
                    
                    while (remaining > 0) {
                        int elements_this_slice = (remaining > MAX_NNZ_PER_SLICE) 
                                                 ? MAX_NNZ_PER_SLICE 
                                                 : remaining;
                        
                        // If we're not at the start of a new slice, create one
                        if (nnz_in_slice > 0) {
                            Aslices[num_slices] = pos;
                            if (nnz_in_slice > max_slice_nnz) {
                                max_slice_nnz = nnz_in_slice;
                            }
                            num_slices++;
                            nnz_in_slice = 0;
                        }
                        
                        // Add elements to this slice
                        pos += elements_this_slice;
                        nnz_in_slice = elements_this_slice;
                        remaining -= elements_this_slice;
                        
                        // If we still have elements remaining, create a new slice
                        if (remaining > 0) {
                            Aslices[num_slices] = pos;
                            if (nnz_in_slice > max_slice_nnz) {
                                max_slice_nnz = nnz_in_slice;
                            }
                            num_slices++;
                            nnz_in_slice = 0;
                        }
                    }
                } else {
                    // Row fits in a slice
                    // Check if adding this row would exceed the slice capacity
                    if (nnz_in_slice + elements_in_current_row > MAX_NNZ_PER_SLICE) {
                        // Start a new slice
                        Aslices[num_slices] = row_start_idx;
                        if (nnz_in_slice > max_slice_nnz) {
                            max_slice_nnz = nnz_in_slice;
                        }
                        num_slices++;
                        nnz_in_slice = elements_in_current_row;
                    } else {
                        // Add to current slice
                        nnz_in_slice += elements_in_current_row;
                    }
                }
            }
            
            // Reset for new row
            row_start_idx = i;
            last_row = current_row;
        }
    }

    // Handle the last row
    elements_in_current_row = values - row_start_idx;
    if (elements_in_current_row > MAX_NNZ_PER_SLICE) {
        // Split the last row across multiple slices
        int remaining = elements_in_current_row;
        int pos = row_start_idx;
        
        while (remaining > 0) {
            int elements_this_slice = (remaining > MAX_NNZ_PER_SLICE) 
                                     ? MAX_NNZ_PER_SLICE 
                                     : remaining;
            
            // If we're not at the start of a new slice, create one
            if (nnz_in_slice > 0 && nnz_in_slice != elements_this_slice) {
                Aslices[num_slices] = pos;
                if (nnz_in_slice > max_slice_nnz) {
                    max_slice_nnz = nnz_in_slice;
                }
                num_slices++;
                nnz_in_slice = 0;
            }
            
            // Add elements to this slice
            pos += elements_this_slice;
            nnz_in_slice = elements_this_slice;
            remaining -= elements_this_slice;
            
            // If we still have elements remaining, create a new slice
            if (remaining > 0) {
                Aslices[num_slices] = pos;
                if (nnz_in_slice > max_slice_nnz) {
                    max_slice_nnz = nnz_in_slice;
                }
                num_slices++;
                nnz_in_slice = 0;
            }
        }
    } else if (nnz_in_slice + elements_in_current_row > MAX_NNZ_PER_SLICE) {
        // Start a new slice for the last row
        Aslices[num_slices] = row_start_idx;
        if (nnz_in_slice > max_slice_nnz) {
            max_slice_nnz = nnz_in_slice;
        }
        num_slices++;
        nnz_in_slice = elements_in_current_row;
    } else {
        // Add to current slice
        nnz_in_slice += elements_in_current_row;
    }

    // Add the final slice endpoint
    Aslices[num_slices] = values;
    if (nnz_in_slice > max_slice_nnz) {
        max_slice_nnz = nnz_in_slice;
    }
    num_slices++;

    // Calculate shared memory size based on max slice size
    int sharedMemSize = max_slice_nnz * sizeof(double) + max_slice_nnz * sizeof(int);
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
    int threadsPerBlock = 256;
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
        // printf("Kernel completed in %fms\n", e_time);
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
    compute_band_gflops(rows, cols, values, avg_time, Acols, estimated_slices);

    fclose(fin);
    
    // Free using cudaFree instead of free
    cudaFree(Arows);
    cudaFree(Acols);
    cudaFree(Avals);
    cudaFree(v);
    cudaFree(C);

    return 0;
}