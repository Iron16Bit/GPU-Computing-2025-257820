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

void matrix_multiplication(int *row_pointer, int *Acols, double *Avals, double *v, double *C, int rows, int cols, int values) {
    for (int i=0; i<rows; i++) {
        int prev_row = row_pointer[i];
        int curr_row = row_pointer[i+1];

        for (int j=prev_row; j<curr_row; j++) {
            C[i] += (Avals[j] * v[Acols[j]]);
        }
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

void init_row_pointer(int *Arows, int *row_pointer, int values, int rows) {
    row_pointer[0] = 0;
    int counter = 0;
    int last_pos = 0;
    for (int i=0; i<rows; i++) {
        for (int j=last_pos; j<values; j++) {
            if (Arows[j] == i) {
                counter += 1;
            } else {
                last_pos = j;
                row_pointer[i+1] = counter;
                break;
            }
        }
    }
    row_pointer[rows] = counter;
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

    // Sort COO
    sort(Arows, Acols, Avals, values);

    // Create CSR's row pointer
    int* row_pointer = (int*)malloc(rows*sizeof(int)+1);
    init_row_pointer(Arows, row_pointer, values, rows);
    free(Arows);

    // Create dense vector
    double *v = malloc(cols*sizeof(double));
    for (int i=0; i<cols; i++) {
        v[i] = 1.0;
    }

    // Compute multiplication
    double *C = (double *)malloc(rows*sizeof(double));
    memset(C, 0, rows*sizeof(double));
    TIMER_DEF(var);
    TIMER_START(var);
    matrix_multiplication(row_pointer, Acols, Avals, v, C, rows, cols, values); 
    TIMER_STOP(var);
    // print_double_array(C, rows);
    printf("[CPU csr] Elapsed time: %f\n", TIMER_ELAPSED(var));

    fclose(fin);

    free(Acols);
    free(Avals);
    free(row_pointer);
    free(v);
    free(C);

    return 0;
}