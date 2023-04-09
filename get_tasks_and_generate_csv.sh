file_name=$1
benchmark_name=${file_name%%-*}
if [ -z "$2" ]; then port_num="6060"; else port_num="$2" fi
path_to_file=$benchmark_name/$file_name
scp -P $port_num debian@pc012170.fit.vutbr.cz:smt-bench/bench/$file_name $path_to_file

git_message=""

if [[ $path_to_file == *"z3-noodler"* ]]; then
	GIT_COMMIT=$(ggrep -m 1 -Po '.{7}(?=-result)' $path_to_file)
	sed -i '' "s/$GIT_COMMIT-result/result/g" $path_to_file
	sed -i '' "s/z3-noodler/z3-noodler-$GIT_COMMIT/g" $path_to_file
	if [[ $path_to_file == *"z3-noodler-underapprox"* ]]; then
		git_message="z3-noodler-underapprox-$GIT_COMMIT on $benchmark_name"
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
