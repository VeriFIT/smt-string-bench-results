#!/bin/zsh

# Fill these out
HOST=""
PORT=""
FILE_PATH_ON_HOST=""

file_name=$1
benchmark_name=${file_name%%-*}
path_to_file=$benchmark_name/$file_name

if ssh -p $PORT $HOST "test -e $FILE_PATH_ON_HOST/$file_name"; then
	scp -P $PORT $HOST:$FILE_PATH_ON_HOST/$file_name $path_to_file
else
	echo "File not found on any server"
	exit 1
fi

git_message=""

if [[ $path_to_file == *"z3-noodler"* ]]; then
	GIT_COMMIT=$(ggrep -m 1 -Po '.{15}(?=-result)' $path_to_file)
	sed -i '' "s/$GIT_COMMIT-result/result/g" $path_to_file
	sed -i '' "s/z3-noodler/z3-noodler-$GIT_COMMIT/g" $path_to_file
	if [[ $path_to_file == *"z3-noodler-underapprox"* ]]; then
		git_message="z3-noodler-$GIT_COMMIT-underapprox on $benchmark_name"
	elif [[ $path_to_file == *"z3-noodler-loop"* ]]; then
		git_message="z3-noodler-$GIT_COMMIT-loop on $benchmark_name"
	elif [[ $path_to_file == *"z3-noodler-nielsen"* ]]; then
		git_message="z3-noodler-$GIT_COMMIT-nielsen on $benchmark_name"
	else
		git_message="z3-noodler-$GIT_COMMIT on $benchmark_name"
	fi
else
	tools="${path_to_file#*to120-}"
	tools="${tools%%-2023*}"
	tools="${tools/z3-trau/z3 trau}"
	tools="${tools//-/, }"
	tools="${tools/z3 trau/z3-trau}"
	git_message="$tools on $benchmark_name"
fi


# ls $benchmark_name/*.tasks | sed "s/-underapprox/underapprox/g" | sort -t- -k 5 | sed "s/underapprox/-underapprox/g" | xargs cat | python3 pyco_proc --csv > $benchmark_name/to120.csv
ls $benchmark_name/*.tasks | sed "s/2023/@/g" | sort -t@ -k 2 | sed "s/@/2023/g" | xargs cat | python3 pyco_proc --csv > $benchmark_name/to120.csv
git add $path_to_file $benchmark_name/to120.csv
git commit -m "$git_message"
