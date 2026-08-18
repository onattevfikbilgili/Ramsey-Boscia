#!/bin/bash

# Check that exactly 3 arguments were provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 n r s"
    exit 1
fi

# Check that all arguments are integers
for arg in "$@"; do
    if ! [[ "$arg" =~ ^-?[0-9]+$ ]]; then
        echo "Error: all arguments must be integers"
        exit 1
    fi
done

# Print arguments at the beginning
echo "===== n=$1, r=$2, s=$3 =====" >> outputs.txt
echo " " >> outputs.txt

# Run every Julia script in the current directory
for script in *.jl; do
    # Skip if there are no .jl files
    [ -e "$script" ] || continue

    echo "===== Running $script =====" >> outputs.txt

    timeout 1h julia "$script" "$1" "$2" "$3" >> outputs.txt 2>&1

    # Record if the script timed out
    if [ $? -eq 124 ]; then
        echo "===== $script TIMED OUT after 1 hour =====" >> outputs.txt
    fi

    echo "===== Finished $script =====" >> outputs.txt
    echo " " >> outputs.txt
    echo " " >> outputs.txt
done
