#!/bin/bash

# Define the size threshold in KB
SIZE_THRESHOLD=102400

# Find files larger than the specified threshold and remove them from Git tracking
find . -type f -size +${SIZE_THRESHOLD}k -print0 | while IFS= read -r -d $'\0' file; do
    echo "Removing large file from tracking: $file"
    git rm --cached "$file"
    # Optionally, echo the file name to a .gitignore to automatically ignore it
    echo "$file" >> .gitignore
done

# Commit changes and update .gitignore
git commit --amend -CHEAD

