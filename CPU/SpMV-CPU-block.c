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

void block_matrix_multiplication(double *M, double* v, double *C, int cols, int row_block, int col_block, int iteration) {
    int c_offset = iteration / (cols/col_block) * row_block;
    int v_offset = iteration % (cols/col_block) * col_block;
    int m_offset = c_offset * col_block + v_offset;

    for (int i=0; i<row_block; i++) {
        for (int j=0; j<1; j++) {
            for (int k=0; k<col_block; k++) {
                C[c_offset + i] += M[m_offset + i * cols + k] * v[v_offset + k];
            }
        }
    }
}

int best_block(int n) {
    for (int i=n-1; i>=0; i--) {
        if (n%i == 0) {
            return i;
        }
    }
}

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
    
    // Block-based approach
    // Find block sizes
    int row_block = best_block(rows);
    int col_block = best_block(cols);

    double *M = (double *)malloc(rows*cols*sizeof(double));
    double *C = (double *)malloc(rows*sizeof(double));
    memset(M, 0, rows*cols*sizeof(double));
    memset(C, 0, rows*sizeof(double));
    TIMER_DEF(var);
    TIMER_START(var);
    // COO -> Matrix
    for(int i=0; i<values; i++) {
        M[Arows[i]*cols+Acols[i]] = Avals[i];
    }
    // Perform block-based matrix multiplication
    for (int i=0; i<(rows*cols)/(row_block*col_block); i++) {
        block_matrix_multiplication(M, v, C, cols, row_block, col_block, i);
    }
    TIMER_STOP(var);
    printf("[CPU block] Elapsed time: %f\n", TIMER_ELAPSED(var));

    fclose(fin);

    free(Arows);
    free(Acols);
    free(Avals);
    free(v);
    free(M);
    free(C);

    return 0;
}