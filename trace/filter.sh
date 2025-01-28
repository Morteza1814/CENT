#!/bin/bash

# Keep FC layers for traces in Model parallel devices

python3 filter.py trace_pipeline_parallel_seq_4096.txt > trace_pim_pipeline_parallel_seq_4096.txt

for resources in 8 16 32 64 128 256
do
    echo "filter.py trace_model_parallel_"$resources"_seq_4096.txt > trace_pim_model_parallel_"$resources"_seq_4096.txt"
    python3 filter.py trace_model_parallel_"$resources"_seq_4096.txt > trace_pim_model_parallel_"$resources"_seq_4096.txt
done