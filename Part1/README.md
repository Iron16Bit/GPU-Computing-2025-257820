# GPU Computing 2025 Project - Student 257820

What follows is a description of the various algorithms used to solve the "Sparse Matrix Dense Vector Multiplication" problem.

## CPU

### Naive Approach

The simplest matrix multiplication algorithm. To perform M x v:
- Iterate through the rows of M
- Iterate through v (a single column vector)
- Compute the vector product

### Block-based Approach

The same as before, but M and v are split into blocks and the multiplication is performed among those smaller blocks.
This implementation is still a sequential one.

### COO Approach

The previous implementations trasformed the COO into a sparse matrix. This one instead directly uses the COO.