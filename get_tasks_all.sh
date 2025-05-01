#!/bin/bash

# Check if a file argument is provided
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <file>"
    exit 1
fi

# Check if the provided file exists
if [[ ! -f $1 ]]; then
    echo "Error: File $1 not found!"
    exit 1
fi

# Read all lines using cat and process each line
for line in $(cat "$1"); do
    # Skip empty lines
    [[ -z "$line" ]] && continue
    echo "$line"
    ./get_tasks_and_generate_csv.sh "$line"
done