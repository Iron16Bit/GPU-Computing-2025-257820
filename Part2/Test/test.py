#!/usr/bin/env python3
# filepath: c:\Users\gille\Desktop\GPU-Computing-2025-257820\Part2\Test\compare_first_line_fixed.py

import sys
import numpy as np
import re

def read_first_line_results(filename):
    """Read numerical results from the first line only, handling BOM if present."""
    try:
        with open(filename, 'r', encoding='utf-8-sig') as f:  # Use 'utf-8-sig' to handle BOM
            first_line = f.readline().strip()
            print(f"DEBUG: Raw first line from '{filename}': {first_line}")  # Debugging line
        
        if not first_line:
            print(f"Warning: First line is empty in '{filename}'.")
            return np.array([])
        
        # Use regex to find all floating point numbers (including scientific notation)
        number_pattern = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?'
        numbers = re.findall(number_pattern, first_line)
        
        # Convert to float with higher precision
        float_numbers = []
        for num_str in numbers:
            try:
                float_numbers.append(float(num_str))
            except ValueError:
                continue
        
        print(f"Found {len(float_numbers)} numbers in first line of '{filename}'")
        print(f"First 5 numbers: {float_numbers[:5]}")
        return np.array(float_numbers, dtype=np.float64)
    
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except Exception as e:
        print(f"Error reading '{filename}': {e}")
        return None

def find_exact_differences(arr1, arr2, file1_name, file2_name):
    """Find exact differences with very strict tolerance"""
    
    if arr1 is None or arr2 is None:
        print("Cannot compare: One or both files could not be read.")
        return
    
    print("="*80)
    print("EXACT COMPARISON RESULTS")
    print("="*80)
    print(f"File 1: {file1_name} ({len(arr1)} elements)")
    print(f"File 2: {file2_name} ({len(arr2)} elements)")
    print()
    
    # Compare lengths
    if len(arr1) != len(arr2):
        print(f"⚠️  Arrays have different lengths!")
        print(f"   {file1_name}: {len(arr1)} elements")
        print(f"   {file2_name}: {len(arr2)} elements")
        min_len = min(len(arr1), len(arr2))
        print(f"   Comparing first {min_len} elements")
        arr1 = arr1[:min_len]
        arr2 = arr2[:min_len]
        print()
    
    # Find exact differences (no tolerance)
    exact_diff_mask = (arr1 != arr2)
    exact_diff_indices = np.where(exact_diff_mask)[0]
    
    # Find differences with tiny tolerance
    tiny_tolerance = 1e-15
    diff = arr1 - arr2
    abs_diff = np.abs(diff)
    tiny_diff_mask = (abs_diff > tiny_tolerance)
    tiny_diff_indices = np.where(tiny_diff_mask)[0]
    
    print(f"Exact differences (no tolerance): {len(exact_diff_indices)}")
    print(f"Differences > {tiny_tolerance}: {len(tiny_diff_indices)}")
    print()
    
    if len(exact_diff_indices) == 0:
        print("✅ ALL ELEMENTS ARE EXACTLY IDENTICAL!")
        print("The arrays are numerically identical.")
        return
    
    # Show first few exact differences
    print("EXACT DIFFERENCES (first 10):")
    print("-" * 90)
    print(f"{'Index':<8} {'File1 Value':<25} {'File2 Value':<25} {'Difference':<20}")
    print("-" * 90)
    
    show_count = min(10, len(exact_diff_indices))
    for i in range(show_count):
        idx = exact_diff_indices[i]
        val1 = arr1[idx]
        val2 = arr2[idx]
        diff_val = val1 - val2
        
        print(f"{idx:<8} {val1:<25.15e} {val2:<25.15e} {diff_val:<20.15e}")
    
    if len(exact_diff_indices) > 10:
        print(f"... and {len(exact_diff_indices) - 10} more exact differences")
    
    # Check if differences are only in very small values
    small_value_threshold = 1e-10
    small_diffs = exact_diff_indices[np.abs(arr2[exact_diff_indices]) < small_value_threshold]
    
    if len(small_diffs) > 0:
        print(f"\nNote: {len(small_diffs)} differences are in very small values (< {small_value_threshold})")
        print("These might be due to floating-point precision or different representations of zero.")

def main():
    # Default file names
    file1 = "output.txt"
    file2 = "testingLog.txt"
    
    # Allow command line arguments
    if len(sys.argv) == 3:
        file1 = sys.argv[1]
        file2 = sys.argv[2]
    elif len(sys.argv) != 1:
        print("Usage: python compare_first_line_fixed.py [file1] [file2]")
        print("Default: compare first line of output.txt and testingLog.txt")
        sys.exit(1)
    
    print(f"Comparing first line results from '{file1}' and '{file2}'...")
    print()
    
    # Read the results from first line only
    arr1 = read_first_line_results(file1)
    arr2 = read_first_line_results(file2)
    
    # Perform exact comparison
    find_exact_differences(arr1, arr2, file1, file2)

if __name__ == "__main__":
    main()