#!/bin/zsh

benchmark_name=$1
ls $benchmark_name/*.tasks | awk '{print substr($0, 1, length($0)-22) "@" substr($0, length($0)-21)}' | sort -t@ -k 2 | sed "s/@//g" | xargs cat | python3 pyco_proc --csv > $benchmark_name/to120.csv
