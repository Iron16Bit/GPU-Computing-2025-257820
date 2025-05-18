# GPU Computing 2025 Project - Student 257820

## Building the project

Just use: ```make``` \
The makefile will download the dataset, extract it and sort the matrices.

## Running the algorithms

To run an algorithm on a specific matrix, use: ```./run.sh``` \
You will be asked to choose the algorithm and the matrix based on those available and based on the chosen algorithm also the number of **BLOCKS** and **THREADS PER BLOCK**. The output of the scheduled job will be found in the *./outputs/* folder called "fileName_jobNumber.txt".

**NB:** In case of error, make sure the *run.sh* file is executable by using ```chmod +x run.sh```

## Dataset Information

These are the [SuiteSparse Matrix Collection](https://sparse.tamu.edu/) matrices used to test the various algorithms.
The links allow you to see the shape of various matrices.

### Diagonal Matrices - GPU
https://sparse.tamu.edu/Bodendiek/CurlCurl_4 \
https://sparse.tamu.edu/Schenk_AFE/af_shell8 \
https://sparse.tamu.edu/Bai/af23560
### Unstructured Matrices - GPU
https://sparse.tamu.edu/MAWI/mawi_201512012345 \
https://sparse.tamu.edu/Belcastro/human_gene2 \
https://sparse.tamu.edu/Mallya/lhr10 \

### Diagonal Matrices - CPU
https://sparse.tamu.edu/CPM/cz5108 \
https://sparse.tamu.edu/Muite/Chebyshev3 \
https://sparse.tamu.edu/Fluorem/DK01R
### Unstructured Matrices - CPU
https://sparse.tamu.edu/Gaertner/pesa \
https://sparse.tamu.edu/Mallya/lhr02 \
https://sparse.tamu.edu/HB/bcsstk08