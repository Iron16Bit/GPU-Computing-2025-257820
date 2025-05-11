#!/bin/bash

mkdir -p Data

# Array of dataset URLs
urls=(
    # Diagonal matrices - GPU
    "https://sparse.tamu.edu/Bodendiek/CurlCurl_4"
    "https://suitesparse-collection-website.herokuapp.com/MM/Schenk_AFE/af_shell8.tar.gz"
    # Diagonal matrices - CPU
    "https://sparse.tamu.edu/Andrews/Andrews"
    "https://sparse.tamu.edu/Bai/af23560"

    # Arrowhead matrices - GPU
    "https://suitesparse-collection-website.herokuapp.com/MM/Rajat/rajat31.tar.gz"
    # Arrowhead matrices - CPU
    "https://sparse.tamu.edu/TKK/cyl6"

    # Unstructured matrices - GPU
    "https://suitesparse-collection-website.herokuapp.com/MM/MAWI/mawi_201512012345.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Belcastro/human_gene2.tar.gz"
    # Unstructured matrices - CPU
    "https://sparse.tamu.edu/AMD/G2_circuit"
    "https://sparse.tamu.edu/HB/cegb2919"
)

# Make sure Data directory exists
mkdir -p Data

for url in "${urls[@]}"; do
    # Get the filename from the URL
    filename=$(basename "$url")
    filepath="Data/$filename"

    # Download the file
    wget "$url" -P Data/

    # Extract the tar.gz file
    tar -xf "$filepath" -C Data/

    # Remove the tar.gz file
    rm "$filepath"

    # Extract the directory name (without .tar.gz)
    dirname="${filename%.tar.gz}"

    # Move the .mtx file to Data/ root
    mv "Data/$dirname/$dirname.mtx" "Data/$dirname.mtx"

    # Remove the extracted directory
    rm -rf "Data/$dirname"
done
