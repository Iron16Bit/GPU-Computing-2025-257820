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

void matrix_multiplication(int *Arows, int *Acols, double *Avals, double *v, double *C, int rows, int cols, int values) {
    // Initialize result vector to zeros (this is already done in main())
    // memset(C, 0, rows * sizeof(double));
    
    int i = 0;
    while (i < values) {
        int row = Arows[i];
        double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
        
        // Process all elements in the current row
        while (i < values && Arows[i] == row) {
            // Process 4 elements at a time if we don't spill to the next row
            if (i + 3 < values && Arows[i+3] == row) {
                sum0 += Avals[i] * v[Acols[i]];
                sum1 += Avals[i+1] * v[Acols[i+1]];
                sum2 += Avals[i+2] * v[Acols[i+2]];
                sum3 += Avals[i+3] * v[Acols[i+3]];
                i += 4;
            } else {
                // Handle one element at a time for the remaining elements of this row
                sum0 += Avals[i] * v[Acols[i]];
                i++;
            }
        }
        
        C[row] += sum0 + sum1 + sum2 + sum3;
    }
}

#define ITERATIONS 51

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

    double *C = (double *)malloc(rows*sizeof(double));
    double tot_time = 0.0;

    first = 1;

    for (int i=0; i<ITERATIONS; i++) {
        memset(C, 0, rows*sizeof(double));
        TIMER_DEF(var);
        TIMER_START(var);
        matrix_multiplication(Arows, Acols, Avals, v, C, rows, cols, values); 

        TIMER_STOP(var);
        // printf("[CPU coo] Elapsed time: %f\n", TIMER_ELAPSED(var));
        if (first == 1) {
            first = 0;
        } else {
            tot_time += TIMER_ELAPSED(var);
        }
    }
    printf("[CPU multiAcc] Average elapsed time: %fms\n", tot_time / (ITERATIONS - 1));
    // print_double_array(C, rows);

    fclose(fin);

    free(Arows);
    free(Acols);
    free(Avals);
    free(v);
    free(C);

    return 0;
}