#!/usr/bin/env bash

FILE_PATH_ON_HOST="../smt-bench/bench"

# Exctracts tool name from the first argument which is assumed
# to be a file path of form
#     benchmark_name-toD*-tool_name-date.tasks
# where D is some digit and date has the form "YYYY-MM-DD-hh-mm".
extract_tool_name() {
    # tool name should start after second occurence of '-'
    local tool_name=$(cut -d- -f3- <<< $1)
    # remove .tasks from the end
    tool_name=${tool_name%.tasks}
    # remove date (should be exactly 17 characters long)
    tool_name=${tool_name:0:-17}
    echo "$tool_name"
}

# Extracts the tool version from the line containing substring ";version-result".
get_tool_version() {
	local line_with_version="$1"
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

	if test -e $FILE_PATH_ON_HOST/$file_name; then
		cp $FILE_PATH_ON_HOST/$file_name $path_to_file
	else
		echo "File $FILE_PATH_ON_HOST/$file_name not found"
		exit 1
	fi

	local git_message=""

	local line_with_version=$(grep -m 1 -- "-result" "$path_to_file")
	if [ -z "$line_with_version" ]; then
		git_message="$tool_name on $benchmark_name"
	else
		local version=$(get_tool_version "$line_with_version")
		sed -i "s/$version-result/result/g" $path_to_file
		sed -i "s/$tool_name;/$tool_name-$version;/g" $path_to_file
		git_message="$tool_name-$version on $benchmark_name"
		mv $path_to_file "${path_to_file//$tool_name/$tool_name-$version}"
		path_to_file="${path_to_file//$tool_name/$tool_name-$version}"
	fi

	#git add $path_to_file
	#git commit -m "$git_message"
}

if [ "$#" -ne 1 ]; then
    echo "ERROR: script expects exactly one argument - name of the .tasks file that should be downloaded and processed"
	exit -1
fi

process_tasks $1
