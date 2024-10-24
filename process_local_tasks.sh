#!/usr/bin/env bash

for line in $(cat $1); do
  echo "Processing $line";
  ./get_local_tasks_and_generate_csv.sh "$line";
done

