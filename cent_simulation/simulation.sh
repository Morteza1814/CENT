python3 run_sim.py --model Llama2-7B --generate_trace --simulate_trace --process_results --update_csv --num_devices 8 --run_simulation_max_workers 80
python3 run_sim.py --model Llama2-13B --generate_trace --simulate_trace --process_results --update_csv --num_devices 20 --run_simulation_max_workers 40
python3 run_sim.py --model Llama2-70B --generate_trace --simulate_trace --process_results --update_csv --num_devices 32 --run_simulation_max_workers 10

python3 run_sim.py --model Llama2-7B --model_parallel --generate_trace --simulate_trace --process_results --update_csv --num_devices 8 --run_simulation_max_workers 80
python3 run_sim.py --model Llama2-13B --model_parallel --generate_trace --simulate_trace --process_results --update_csv --num_devices 20 --run_simulation_max_workers 80
python3 run_sim.py --model Llama2-70B --model_parallel --generate_trace --simulate_trace --process_results --update_csv --num_devices 32 --run_simulation_max_workers 80
