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

void block_matrix_multiplication(int* Arows, int* Acols, double* Avals, double* v, double *C, int values, int BLOCK_SIZE, int iteration) {
    // Calculate starting positions for the block
    int start = iteration * BLOCK_SIZE;
    int end = (iteration + 1) * BLOCK_SIZE;
    
    // Process the block
    for (int i=start; i<end; i++) {
        if (i >= values) {
            break;
        }
        int row = Arows[i];
        int col = Acols[i];
        
        C[row] += Avals[i] * v[col];
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

    float total_time = 0.0;

    for (int i=0; i<ITERATIONS; i++) {
        memset(C, 0, rows*sizeof(double));
        TIMER_DEF(var);
        TIMER_START(var);

        int total_blocks = (values + BLOCK_SIZE - 1) / BLOCK_SIZE;

        for (int i = 0; i < total_blocks; i++) {
            block_matrix_multiplication(Arows, Acols, Avals, v, C, values, BLOCK_SIZE, i);
        }

        TIMER_STOP(var);
        // printf("[CPU block] Elapsed time: %fms\n", TIMER_ELAPSED(var));
        if (i != 0) {
            total_time += TIMER_ELAPSED(var);
        }
    }
    printf("[CPU block-coo] Average time: %fms\n", total_time / (ITERATIONS-1));
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