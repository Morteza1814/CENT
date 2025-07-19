#!/bin/bash

#SBATCH --ntasks=1            
#SBATCH --cpus-per-task=32    
#SBATCH --job-name=run_sim_7B_32sqgap
#SBATCH --partition=cpu
#SBATCH --mem=100000
#SBATCH --time=2-00:00:00
#SBATCH --output=run_sim_7B_32sqgap.txt
#SBATCH --error=run_sim_7B_32sqgap.txt

python3 run_sim.py --model Llama2-7B --generate_trace --simulate_trace --process_results --update_csv \
--num_devices 8 --run_simulation_max_workers 32 --generate_trace_max_workers 32 --seqlen_gap 32 \
--simulation_result_path simulation_results_7B_32sqgap.csv 

python3 run_sim.py --model Llama2-7B --generate_trace --simulate_trace --process_results --update_csv \
--num_devices 32 --run_simulation_max_workers 32 --generate_trace_max_workers 32 --seqlen_gap 32 \
--simulation_result_path simulation_results_7B_32sqgap.csv 

# Model Parallel
python3 run_sim.py --model Llama2-7B --model_parallel --generate_trace --simulate_trace --process_results --update_csv \
--num_devices 8 --run_simulation_max_workers 32 --generate_trace_max_workers 32 --seqlen_gap 32 \
--simulation_result_path simulation_results_7B_32sqgap.csv 

python3 run_sim.py --model Llama2-7B --model_parallel --generate_trace --simulate_trace --process_results --update_csv \
--num_devices 32 --run_simulation_max_workers 32 --generate_trace_max_workers 32 --seqlen_gap 32 \
--simulation_result_path simulation_results_7B_32sqgap.csv 
