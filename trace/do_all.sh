cp /data1/yufenggu/llama-cpu/trace/run.sh ./
cp /data1/yufenggu/llama-cpu/trace/compile.sh ./
cp /data1/yufenggu/llama-cpu/trace/compile.py ./
# bash run.sh
bash compile.sh &> result.txt
python compile.py ./result.txt &> compiled_results.txt
