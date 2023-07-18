#!/bin/zsh

# Fill these out
HOST=""
PORT=""
FILE_PATH_ON_HOST=""

# Exctracts tool name from the first argument which is assumed
# to be a file path of form
#     benchmark_name-to120-tool_name-date.tasks
# where date has the form "YYYY-MM-DD-hh-mm".
extract_tool_name() {
    # tool name should start after "*-to120-"
    local tool_name=${1#*to120-}
    # remove .tasks from the end
    tool_name=${tool_name%.tasks}
    # remove date (should be exactly 17 characters long)
    tool_name=${tool_name:0:-17}
    
    echo "$tool_name"
}

# Extracts the tool version from the .tasks file given as an argument.
# It assumes that the .tasks file contains at least one line
# containing substring ";version-result".
get_tool_version() {
	local line_with_version=$(grep -m 1 "-result" "$1")
	line_with_version=${line_with_version%-result*}
	line_with_version=${line_with_version##*;}
	echo ${line_with_version%;}
}

# Takes .tasks file as an argument which is downloaded from
# the server, processed, new .csv is created and git commited.
process_tasks() {
	local file_name=$1
	local tool_name=$(extract_tool_name "$file_name")
	local benchmark_name=${file_name%%-*}
	local path_to_file=$benchmark_name/$file_name

	if ssh -p $PORT $HOST "test -e $FILE_PATH_ON_HOST/$file_name"; then
		scp -P $PORT $HOST:$FILE_PATH_ON_HOST/$file_name $path_to_file
	else
		echo "File not found on any server"
		exit 1
	fi

	local git_message=""

	if grep -q "-result" "$path_to_file"; then
		local version=$(get_tool_version $path_to_file)
		sed -i '' "s/$version-result/result/g" $path_to_file
		sed -i '' "s/$tool_name/$tool_name-$version/g" $path_to_file
		git_message="$tool_name-$version on $benchmark_name"
	else
		git_message="$tool_name on $benchmark_name"
	fi


	# Take all the .tasks files for the given benchmark, sort them by dates and give that to pyco_proc, so that newest benchmarks are always at the end of csv.
	ls $benchmark_name/*.tasks | awk '{print substr($0, 1, length($0)-22) "@" substr($0, length($0)-21)}' | sort -t@ -k 2 | sed "s/@//g" | xargs cat | python3 pyco_proc --csv > $benchmark_name/to120.csv

	git add $path_to_file $benchmark_name/to120.csv
	git commit -m "$git_message"
}

process_tasks $1
