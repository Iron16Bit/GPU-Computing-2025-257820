#!/bin/bash

mkdir -p Data

# Array of dataset URLs
urls=(
    "https://suitesparse-collection-website.herokuapp.com/MM/HB/1138_bus.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/HB/bcsstk18.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/HB/bcsstm25.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/HB/gemat11.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/gridgena.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/VanVelzen/Zd_Jac3_db.tar.gz"
    # "https://suitesparse-collection-website.herokuapp.com/MM/MAWI/mawi_201512012345.tar.gz"
    # "https://suitesparse-collection-website.herokuapp.com/MM/Janna/ML_Geer.tar.gz"
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
