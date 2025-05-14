#!/bin/bash

# Output files
all_file="all_results.txt"
best_bandwidth_file="best_bandwidth.txt"
best_flops_file="best_flops.txt"

# Clear output files
> "$all_file"
> "$best_bandwidth_file"
> "$best_flops_file"

# Temp files
tmp_bandwidth=$(mktemp)
tmp_flops=$(mktemp)

# Headers
printf "%-20s %-30s %-6s %-6s %-10s %-12s %-10s\n" \
  "Algorithm" "Matrix" "Blocks" "TPB" "Time(ms)" "Bandwidth" "FLOPS" >> "$all_file"

# Process input files
for file in outputs/*.txt; do
    algorithm=$(basename "$file" | cut -d'_' -f1)

    # Extract and normalize matrix name
    matrix_full=$(grep -oP 'Running on matrix: \K.*' "$file")
    matrix=${matrix_full%%_sorted*}
    matrix=${matrix:-N/A}

    config_line=$(grep "Using configuration" "$file")
    blocks=""
    tpb=""

    if [[ $config_line =~ ([0-9]+)\ threads\ per\ block ]]; then
        tpb="${BASH_REMATCH[1]}"
    fi
    if [[ $config_line =~ ([0-9]+)\ blocks ]]; then
        blocks="${BASH_REMATCH[1]}"
    fi

    time=$(grep -oP 'Average (elapsed )?time: \K[0-9.]+' "$file")
    bandwidth=$(grep -oP 'Bandwidth: \K[0-9.]+' "$file")
    flops=$(grep -oP 'FLOPS: \K[0-9.]+' "$file")

    blocks=${blocks:-N/A}
    tpb=${tpb:-N/A}
    time=${time:-N/A}
    bandwidth=${bandwidth:-0}
    flops=${flops:-0}

    # Append to all_results
    printf "%-20s %-30s %-6s %-6s %-10s %-12s %-10s\n" \
        "$algorithm" "$matrix" "$blocks" "$tpb" "$time" "$bandwidth" "$flops" >> "$all_file"

    # Temp entries for best-of
    echo "$algorithm|$matrix|$bandwidth" >> "$tmp_bandwidth"
    echo "$algorithm|$matrix|$flops" >> "$tmp_flops"
done

# Best bandwidth
{
    printf "%-20s %-30s %-12s\n" "Algorithm" "Matrix" "Bandwidth"
    awk -F'|' '
    {
        key = $1 "|" $2
        if (!(key in best) || $3+0 > val[key]) {
            best[key] = $0
            val[key] = $3+0
        }
    }
    END {
        for (k in best) {
            split(best[k], f, "|")
            printf "%-20s %-30s %-12s\n", f[1], f[2], f[3]
        }
    }' "$tmp_bandwidth"
} > "$best_bandwidth_file"

# Best FLOPS
{
    printf "%-20s %-30s %-10s\n" "Algorithm" "Matrix" "FLOPS"
    awk -F'|' '
    {
        key = $1 "|" $2
        if (!(key in best) || $3+0 > val[key]) {
            best[key] = $0
            val[key] = $3+0
        }
    }
    END {
        for (k in best) {
            split(best[k], f, "|")
            printf "%-20s %-30s %-10s\n", f[1], f[2], f[3]
        }
    }' "$tmp_flops"
} > "$best_flops_file"

# Cleanup
rm "$tmp_bandwidth" "$tmp_flops"
