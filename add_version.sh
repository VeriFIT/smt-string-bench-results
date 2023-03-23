GIT_COMMIT=$(ggrep -m 1 -Po '.{7}(?=-result)' $1)
sed -i '' "s/$GIT_COMMIT-result/result/g" $1
sed -i '' "s/z3-noodler/z3-noodler-$GIT_COMMIT/g" $1
