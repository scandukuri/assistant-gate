#!/bin/bash

# Define the size threshold
SIZE_THRESHOLD=100M

# Find files larger than the threshold, recursively, and reset them using Git
find . -type f -size +$SIZE_THRESHOLD | while read file; do
    # Reset the file with Git
    git reset "$file"
    echo "Reset $file"
done
