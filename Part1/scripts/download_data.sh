#!/bin/bash

mkdir -p Data

# Array of dataset URLs
urls=(
    # Diagonal matrices - GPU
    "https://suitesparse-collection-website.herokuapp.com/MM/Bodendiek/CurlCurl_4.tar.gz"       #* OK
    "https://suitesparse-collection-website.herokuapp.com/MM/Schenk_AFE/af_shell8.tar.gz"       #* OK
    # Diagonal matrices - CPU
    "https://suitesparse-collection-website.herokuapp.com/MM/GHS_indef/spmsrtls.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Bai/af23560.tar.gz"                #* OK

    # Arrowhead matrices - GPU
    "https://suitesparse-collection-website.herokuapp.com/MM/Rajat/rajat31.tar.gz"              #* OK
    # Arrowhead matrices - CPU
    "https://suitesparse-collection-website.herokuapp.com/MM/TKK/cyl6.tar.gz"                   #* OK

    # Unstructured matrices - GPU
    "https://suitesparse-collection-website.herokuapp.com/MM/MAWI/mawi_201512012345.tar.gz"     #* OK
    "https://suitesparse-collection-website.herokuapp.com/MM/Belcastro/human_gene2.tar.gz"      #* OK
    # Unstructured matrices - CPU
    "https://suitesparse-collection-website.herokuapp.com/MM/Mallya/lhr10.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/GHS_indef/cvxqp3.tar.gz"           #* OK
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
