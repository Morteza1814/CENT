for f in trace*.txt
do
	# log_file="${f/txt/log}"
	# echo "Processing $f file...";
	# echo "Running /data1/yufenggu/ramulator2/build/ramulator2 -f /data1/yufenggu/ramulator2/test/example.yaml -t $f &> $log_file";
	/data1/yufenggu/llama-cpu/ramulator2/build/ramulator2 -f /data1/yufenggu/llama-cpu/ramulator2/test/example.yaml -t $f 2>&1 | grep '^[^\[]' &> $f.log &
done
