# GPU Computing 2025 Project - Student 257820

## Building the project

Just use: ```make``` \
The makefile will download the dataset, extract it and sort the matrices.

## Running the algorithms

To run an algorithm on a specific matrix, use: ```./run.sh``` \
You will be asked to choose the algorithm and the matrix based on those available and based on the chosen algorithm also the number of **BLOCKS** and **THREADS PER BLOCK**. The output of the scheduled job will be found in the *./outputs/* folder called "fileName_jobNumber.txt".

**NB:** in case of error, make sure the *run.sh* file is executable by using ```chmod +x run.sh```

## Dataset Information

These are the [SuiteSparse Matrix Collection](https://sparse.tamu.edu/) matrices used to test the various algorithms.
The links allow you to see the shape of various matrices.

### Diagonal Matrices
https://sparse.tamu.edu/Bodendiek/CurlCurl_4 \
https://sparse.tamu.edu/Schenk_AFE/af_shell8 \
https://sparse.tamu.edu/GHS_indef/spmsrtls \
https://sparse.tamu.edu/Bai/af23560

### Arrowhead Matrices
https://sparse.tamu.edu/Rajat/rajat31 \
https://sparse.tamu.edu/TKK/cyl6

### Unstructured Matrices
https://sparse.tamu.edu/MAWI/mawi_201512012345 \
https://sparse.tamu.edu/Belcastro/human_gene2 \
https://sparse.tamu.edu/Mallya/lhr10 \
https://sparse.tamu.edu/GHS_indef/cvxqp3

### Unstr
https://sparse.tamu.edu/Gaertner/pesa \
https://sparse.tamu.edu/Mallya/lhr02 \
https://sparse.tamu.edu/HB/bcsstk08
### Diagonal
https://sparse.tamu.edu/CPM/cz5108 \
https://sparse.tamu.edu/Muite/Chebyshev3 \
https://sparse.tamu.edu/Fluorem/DK01R