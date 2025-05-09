#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "my_time_lib.h"

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

void block_matrix_multiplication(double *M, double* v, double *C, int cols, int rows, int BLOCK_SIZE, int iteration) {
    // Calculate how many blocks we have per row
    int blocks_per_row = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Get block row and column indices
    int block_row = iteration / blocks_per_row;
    int block_col = iteration % blocks_per_row;
    
    // Calculate starting positions for the block
    int row_start = block_row * BLOCK_SIZE;
    int col_start = block_col * BLOCK_SIZE;
    
    // Process the block
    for (int i = 0; i < BLOCK_SIZE; i++) {
        if (row_start + i >= rows) {
            continue;
        }
        double sum = 0.0;
        for (int k = 0; k < BLOCK_SIZE; k++) {
            if (col_start + k >= cols) {
                continue;
            }
            sum += M[(row_start + i) * cols + (col_start + k)] * v[col_start + k];
        }
        
        // Add to the result vector
        C[row_start + i] += sum;
    }
}

int ITERATIONS = 51;
int BLOCK_SIZE = 128;

int main(int argc, char *argv[]) {
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
                
                Arows = (int *)malloc(values*sizeof(int));
                Acols = (int *)malloc(values*sizeof(int));
                Avals = (double *)malloc(values*sizeof(double));
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
    double *v = malloc(cols*sizeof(double));
    for (int i=0; i<cols; i++) {
        v[i] = 1.0;
    }

    double *M = (double *)malloc(rows*cols*sizeof(double));
    double *C = (double *)malloc(rows*sizeof(double));
    memset(M, 0, rows*cols*sizeof(double));

    // COO -> Matrix
    for(int i=0; i<values; i++) {
        M[Arows[i]*cols+Acols[i]] = Avals[i];
    }

    float total_time = 0.0;

    for (int i=0; i<ITERATIONS; i++) {
        memset(C, 0, rows*sizeof(double));
        TIMER_DEF(var);
        TIMER_START(var);

        int blocks_per_row = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int blocks_per_col = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int total_blocks = blocks_per_row * blocks_per_col;

        for (int i = 0; i < total_blocks; i++) {
            block_matrix_multiplication(M, v, C, cols, rows, BLOCK_SIZE, i);
        }

        TIMER_STOP(var);
        // printf("[CPU block] Elapsed time: %fms\n", TIMER_ELAPSED(var));
        if (i != 0) {
            total_time += TIMER_ELAPSED(var);
        }
    }
    printf("[CPU block] Average time: %fms\n", total_time / (ITERATIONS-1));
    // print_double_array(C, rows);

    fclose(fin);

    free(Arows);
    free(Acols);
    free(Avals);
    free(v);
    free(M);
    free(C);

    return 0;
}