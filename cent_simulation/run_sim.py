import os
import gc
import math
import pandas as pd
import threading
import argparse
import subprocess
import concurrent.futures
from cxl_latency import llama_latency, gpt_latency, vector_latency
from cellar_power_calculator import DRAM_POWER, ACCEL_CYCLE, ACCEL_POWER, SRAM_POWER, CTRL_POWER, commands, isrs, power_calculator, command_processor

def get_args():
    parser = argparse.ArgumentParser('run_scripts.py')
    parser.add_argument("--num_channels", type=int, help="Number of channels per device", default=32)
    parser.add_argument("--num_devices", type=int, help="Number of CXL devices", default=32)
    parser.add_argument("--reuse_size", type=int, help="GB reuse size, depending on register number", default=32)
    parser.add_argument("--generate_trace_max_workers", type=int, help="maximum concurrent threads to generate traces, limited by memory", default=20)
    parser.add_argument("--run_simulation_max_workers", type=int, help="maximum concurrent threads to generate traces, limited by memory", default=4)
    parser.add_argument("--model", choices=["Llama2-7B", "Llama2-13B", "Llama2-70B"], help="LLM Model", required=True)
    parser.add_argument("--generate_trace", action="store_true", help="Generate traces")
    parser.add_argument("--simulate_trace", action="store_true", help="Simulate traces")
    parser.add_argument("--process_results", action="store_true", help="Process results")
    parser.add_argument("--update_csv", action="store_true", help="Update results to csv file")
    parser.add_argument("--simulation_result_path", type=str, help="Path to the result file", default="simulation_results.csv")
    parser.add_argument("--process_throughputs", action="store_true", help="average throughputs for various seqlen")
    parser.add_argument("--final_result_path", type=str, help="Path to the final result file", default="final_results.csv")
    parser.add_argument("--phase", choices=["end2end", "prefill", "decoding"], help="Phase of the model", default="end2end")
    parser.add_argument("--prefill", type=int, help="Prefill length", default=512)
    parser.add_argument("--decoding", type=int, help="Decoding length", default=3584)
    parser.add_argument("--seqlen_gap", type=int, help="Gap between sequence lengths", default=128)
    parser.add_argument("--model_parallel", action="store_true", help="Apply model parallelism, FC_devices parameter is needed")
    parser.add_argument("--FC_devices", type=int, help="Number of devices for FC layer", default=32)
    args = parser.parse_args()
    return args

args = get_args()


KILO = 1000
MEGA = 1000000
GIGA = 1000000000
FREQ = 2.00 * GIGA
WORD_SIZE = 256
tRC = 44.5
tBL = 1.25
tCCDL = 1.0

RV_COUNT = 8
# Latency of 1 SIMD operation
SB_RD_CYCLE = 1.00
SB_WR_CYCLE = 1.00
EXP_LANE_CYCLE = 11.00
RV_RMSNorm_CYCLE = 26.00
RV_ROTEmbed_CYCLE = 3.00 / RV_COUNT
RV_SFT_CYCLE_PIPELINE = 16.00 * SB_WR_CYCLE + 2.00 / RV_COUNT + 1.00 * SB_RD_CYCLE
RV_SFT_CYCLE_SINGLE = 16.00 * SB_WR_CYCLE + 2.00 + 1.00 * SB_RD_CYCLE

PCIE_ENERGY = 4.4

for accel_name in ["RED", "EXP", "VEC", "CTR"]:
    ACCEL_POWER[accel_name]["DYN"] = float(ACCEL_POWER[accel_name]["SWITCH"] + ACCEL_POWER[accel_name]["INT"])
    ACCEL_POWER[accel_name]["STT"] = float(ACCEL_POWER[accel_name]["LEAK"]) / float(GIGA)

InOut_latency = 0.15        # Top-K sampling on CPU
n_heads = {"Llama2-7B": 32, "Llama2-13B": 40, "Llama2-70B": 64}
gqa_factor = {"Llama2-7B": 1, "Llama2-13B": 1, "Llama2-70B": 8}
embedding_size = {"Llama2-7B": 4096, "Llama2-13B": 5120, "Llama2-70B": 8192, "GPT3-175B": 12288, "GPT3-175B-TP-8": 12288, "OPT-66B": 9216}
ffn_size = {"Llama2-7B": 11008, "Llama2-13B": 13824, "Llama2-70B": 28672, "GPT3-175B": 12288*4, "GPT3-175B-TP-8": 12288*4, "OPT-66B": 9216*4}
TransformerBlock_number = {"Llama2-7B": 32, "Llama2-13B": 40, "Llama2-70B": 80}
minimal_channel_per_block = {"Llama2-7B": 5, "Llama2-13B": 8, "Llama2-70B": 6}
pipeline_parallel_mode_list = ["pipeline_parallel", "pipeline_parallel_embedding"]
model_parallel_mode_list = ["model_parallel", "model_parallel_embedding", "model_parallel_FC"]
if args.model == "GPT3-175B":
    model = "--GPT3-175B"
elif args.model == "Llama2-70B" or "Llama3" in args.model:
    model = "--Llama-GQA"
elif "Llama2" in args.model:
    model = "--Llama"

seqlen_list = [i * args.seqlen_gap for i in range(1, (args.prefill + args.decoding) // args.seqlen_gap + 1)]
# seqlen_list = [1024, 2048, 3072, 4096]
    
for mode in pipeline_parallel_mode_list:
    subprocess.run(["mkdir", "-p", f"../trace/{args.num_channels}_channels_per_device/{mode}/{args.model}"])

def generate_trace(args):
    commands_generate_traces = []
    blocks_per_device = (TransformerBlock_number[args.model] - 1) // args.num_devices + 1
    channels_per_block = args.num_channels // blocks_per_device
    
    seqlen = args.prefill + args.decoding
    if args.model_parallel:
        if not os.path.exists(f"../trace/{args.num_channels}_channels_per_device/model_parallel_embedding/{args.model}/trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt"):
            commands_generate_traces.append(["python3", "function_sim.py", model, "--n_heads", str(n_heads[args.model]), "--ffn_dim", str(ffn_size[args.model]), "--embedding", "--only-trace", "--num-channels", str(args.num_channels), "--FC-devices", str(args.FC_devices), "--model-parallel", "--seqlen", str(seqlen), "--op-trace", "--GEMV", "reuse-GB", "--reuse-size", str(args.reuse_size), "--trace-file", f"../trace/{args.num_channels}_channels_per_device/model_parallel_embedding/{args.model}/trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt"])
    else:
        if not os.path.exists(f"../trace/{args.num_channels}_channels_per_device/pipeline_parallel_embedding/{args.model}/trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt"):
            commands_generate_traces.append(["python3", "function_sim.py", model, "--n_heads", str(n_heads[args.model]), "--ffn_dim", str(ffn_size[args.model]), "--embedding", "--only-trace", "--num-channels", str(args.num_channels), "--channels-per-block", str(channels_per_block), "--pipeline-parallel", "--multi-tb-per-device", "--seqlen", str(seqlen), "--op-trace", "--GEMV", "reuse-GB", "--reuse-size", str(args.reuse_size), "--trace-file", f"../trace/{args.num_channels}_channels_per_device/pipeline_parallel_embedding/{args.model}/trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt"])

    for seqlen in seqlen_list:
        
        if channels_per_block < minimal_channel_per_block[args.model]:
            raise ValueError(f"Channels per block {channels_per_block} is less than minimal channel per block {minimal_channel_per_block[args.model]}")

        if args.model_parallel:
            if not os.path.exists(f"../trace/{args.num_channels}_channels_per_device/model_parallel/{args.model}/trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt"):
                commands_generate_traces.append(["python3", "function_sim.py", model, "--n_heads", str(n_heads[args.model]), "--ffn_dim", str(ffn_size[args.model]), "--only-trace", "--num-channels", str(args.num_channels), "--FC-devices", str(args.FC_devices), "--model-parallel", "--seqlen", str(seqlen), "--op-trace", "--GEMV", "reuse-GB", "--reuse-size", str(args.reuse_size), "--trace-file", f"../trace/{args.num_channels}_channels_per_device/model_parallel/{args.model}/trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt"])
            if not os.path.exists(f"../trace/{args.num_channels}_channels_per_device/model_parallel_FC/{args.model}/trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt"):
                commands_generate_traces.append(["python3", "function_sim.py", model, "--n_heads", str(n_heads[args.model]), "--ffn_dim", str(ffn_size[args.model]), "--only-FC", "--only-trace", "--num-channels", str(args.num_channels), "--FC-devices", str(args.FC_devices), "--model-parallel", "--seqlen", str(seqlen), "--op-trace", "--GEMV", "reuse-GB", "--reuse-size", str(args.reuse_size), "--trace-file", f"../trace/{args.num_channels}_channels_per_device/model_parallel_FC/{args.model}/trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt"])
        else:
            if not os.path.exists(f"../trace/{args.num_channels}_channels_per_device/pipeline_parallel/{args.model}/trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt"):
                commands_generate_traces.append(["python3", "function_sim.py", model, "--n_heads", str(n_heads[args.model]), "--ffn_dim", str(ffn_size[args.model]), "--only-trace", "--num-channels", str(args.num_channels), "--channels-per-block", str(channels_per_block), "--pipeline-parallel", "--multi-tb-per-device", "--seqlen", str(seqlen), "--op-trace", "--GEMV", "reuse-GB", "--reuse-size", str(args.reuse_size), "--trace-file", f"../trace/{args.num_channels}_channels_per_device/pipeline_parallel/{args.model}/trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.generate_trace_max_workers) as executor:
        futures = [executor.submit(subprocess.run, cmd) for cmd in commands_generate_traces]
        for future in concurrent.futures.as_completed(futures):
            future.result()

def run_command(command, log_file):
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    filtered_output = "\n".join(line for line in result.stdout.splitlines() if not line.startswith('['))
    with open(log_file, "w") as log:
        log.write(filtered_output)

def simulate_trace(args):
    commands_simulate_traces = []

	# ../ramulator2/build/ramulator2 -f ../ramulator2/test/example.yaml -t ../trace/48_channels_per_device/pipeline_parallel/Llama2-7B/trace_16_channels_per_block_seqlen_{seqlen}.txt 2>&1 | grep '^[^\[]' &> ../trace/48_channels_per_device/pipeline_parallel/Llama2-7B/trace_16_channels_per_block_seqlen_{seqlen}.txt.log

    blocks_per_device = (TransformerBlock_number[args.model] - 1) // args.num_devices + 1
    channels_per_block = args.num_channels // blocks_per_device
    
    for seqlen in seqlen_list:
        if args.model_parallel:

            if not os.path.exists(f"../trace/{args.num_channels}_channels_per_device/model_parallel_embedding/{args.model}/trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt.log"):
                trace_file = f"../trace/{args.num_channels}_channels_per_device/model_parallel_embedding/{args.model}/trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt"
            log_file = f"../trace/{args.num_channels}_channels_per_device/model_parallel_embedding/{args.model}/trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt.log"
            command = f"../ramulator2/build/ramulator2 -f ../ramulator2/test/example.yaml -t {trace_file}"
            commands_simulate_traces.append((command, log_file))

            for mode in ["model_parallel", "model_parallel_FC"]:
                if not os.path.exists(f"../trace/{args.num_channels}_channels_per_device/{mode}/{args.model}/trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt.log"):
                    trace_file = f"../trace/{args.num_channels}_channels_per_device/{mode}/{args.model}/trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt"
                log_file = f"../trace/{args.num_channels}_channels_per_device/{mode}/{args.model}/trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt.log"
                command = f"../ramulator2/build/ramulator2 -f ../ramulator2/test/example.yaml -t {trace_file}"
                commands_simulate_traces.append((command, log_file))
        else:

            if not os.path.exists(f"../trace/{args.num_channels}_channels_per_device/pipeline_parallel_embedding/{args.model}/trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt.log"):
                trace_file = f"../trace/{args.num_channels}_channels_per_device/pipeline_parallel_embedding/{args.model}/trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt"
                log_file = f"../trace/{args.num_channels}_channels_per_device/pipeline_parallel_embedding/{args.model}/trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt.log"
                command = f"../ramulator2/build/ramulator2 -f ../ramulator2/test/example.yaml -t {trace_file}"
                commands_simulate_traces.append((command, log_file))

            for mode in ["pipeline_parallel"]:
                if not os.path.exists(f"../trace/{args.num_channels}_channels_per_device/{mode}/{args.model}/trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt.log"):
                    trace_file = f"../trace/{args.num_channels}_channels_per_device/{mode}/{args.model}/trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt"
                    log_file = f"../trace/{args.num_channels}_channels_per_device/{mode}/{args.model}/trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt.log"
                    command = f"../ramulator2/build/ramulator2 -f ../ramulator2/test/example.yaml -t {trace_file}"
                    commands_simulate_traces.append((command, log_file))

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.run_simulation_max_workers) as executor:
        futures = [executor.submit(run_command, cmd, log) for cmd, log in commands_simulate_traces]
        for future in concurrent.futures.as_completed(futures):
            future.result()

def process_results(args):
    for mode in pipeline_parallel_mode_list:
        compile_dir = f"../trace/{args.num_channels}_channels_per_device/{mode}/{args.model}/"
        subprocess.run(["cp", "../trace/compile.sh", compile_dir])
        subprocess.run(["cp", "../trace/compile.py", compile_dir])

        # Run compile.sh and write output to result.txt
        result = subprocess.run(["bash", "compile.sh"], cwd=compile_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        with open(f"{compile_dir}/result.txt", "w") as result_file:
            result_file.write(result.stdout)
            result_file.write(result.stderr)

        # Run compile.py and write output to compiled_results.txt
        result = subprocess.run(["python3", "compile.py", "./result.txt"], cwd=compile_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        with open(f"{compile_dir}/compiled_results.txt", "w") as compiled_results_file:
            compiled_results_file.write(result.stdout)
            compiled_results_file.write(result.stderr)

def calculate_acc_latency(args, seqlen):
    latency = {}
    GQA_factor = 1.00 + 1.00 / gqa_factor[args.model]
    latency["RMSNorm_latency"] =  embedding_size[args.model] / 16.00 / 16.00 / args.num_channels * ACCEL_CYCLE["VEC"]    # EMB /16.00 /16.00 ADD
    latency["RMSNorm_latency"] += SB_RD_CYCLE + SB_WR_CYCLE + 1.00                              # 1 RED
    latency["RMSNorm_latency"] += RV_RMSNorm_CYCLE                                              # 1 RISCV
    latency["RMSNorm_latency"] = float(2.00 * latency["RMSNorm_latency"]) / float(FREQ / KILO)
    latency["Softmax_latency"] =  seqlen * n_heads[args.model] / 16.00 / args.num_channels * ACCEL_CYCLE["EXP"]        # TOK*HEAD /16.00 EXP
    latency["Softmax_latency"] += seqlen * n_heads[args.model] / 16.00 / args.num_channels * ACCEL_CYCLE["VEC"]        # TOK*HEAD /16.00 ADD
    latency["Softmax_latency"] += n_heads[args.model] * 1.00 * SB_RD_CYCLE                                     # HEAD RED
    latency["Softmax_latency"] += n_heads[args.model] * RV_SFT_CYCLE_PIPELINE                                  # HEAD RISCV
    latency["Softmax_latency"] = float(latency["Softmax_latency"]) / float(FREQ / KILO)
    latency["RotEmbed_latency"] = embedding_size[args.model] * RV_ROTEmbed_CYCLE                                 # EMB RISCV
    latency["RotEmbed_latency"] = float(GQA_factor * latency["RotEmbed_latency"]) / float(FREQ / KILO)
    return latency

def update_csv(args):

    embedding_latency = {}
    embedding_compile_dir = f"../trace/{args.num_channels}_channels_per_device/pipeline_parallel_embedding/{args.model}/"
    with open(f"{embedding_compile_dir}/compiled_results.txt", "r") as compiled_results_file:
        lines = compiled_results_file.readlines()
        for line in lines:
            filename, latency = line.split()[0], line.split()[1]
            channels_per_block = int(filename.split('_')[1])
            embedding_latency[channels_per_block] = float(latency)
    
    '''
    Model: Llama2-7B, Llama2-13B, Llama2-70B
    Channels per device: 24, 32, 40, 48
    Channels per block: 32: {5, 7, 8, 10, 15, 16}
    '''
    if os.path.exists(args.simulation_result_path):
        results_df = pd.read_csv(args.simulation_result_path)
    else:
        columns = ['Model', 'Device number', 'Pipeline parallelism', 'Channels per device', 'Channels per block', 'Sequence length', 'PIM latency', 'CXL latency', 'Acc latency', 'TransformerBlock latency', 'Embedding latency', 'Token latency', 'Throughput', 'Token energy', 'Total power', 'Device utilization']
        results_df = pd.DataFrame(columns=columns)

    for seqlen in seqlen_list:
        pp = TransformerBlock_number[args.model]
        pp_per_device = (pp - 1) // args.num_devices + 1
        blocks_per_device = pp_per_device * (TransformerBlock_number[args.model] // pp)
        channels_per_block = args.num_channels // blocks_per_device
        if channels_per_block < minimal_channel_per_block[args.model]:
            continue
        utilized_devices = (TransformerBlock_number[args.model] - 1) // blocks_per_device + 1
        PCIe_lanes = 144 // args.num_devices
        folder = "pipeline_parallel" if not args.model_parallel else "model_parallel"
        path = f"../trace/{args.num_channels}_channels_per_device/{folder}/{args.model}/trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt.log"
        stats = command_processor(path)
        pim_latency = stats["latency"]
        if args.model_parallel:
            if "Llama" in args.model:
                latency = llama_latency([embedding_size[args.model], ffn_size[args.model]], args.PCIe_lanes, args.FC_devices, args.num_devices)
                print(latency)
            else:
                gpt_latency([embedding_size[args.model], ffn_size[args.model]], args.PCIe_lanes, args.FC_devices, args.num_devices)
        else:
            cxl_latency = vector_latency(embedding_size[args.model], PCIe_lanes)
        acc_latency_dict = calculate_acc_latency(args, seqlen)
        acc_latency = (acc_latency_dict["RMSNorm_latency"] + acc_latency_dict["Softmax_latency"] + acc_latency_dict["RotEmbed_latency"]) * blocks_per_device
        transformer_block_latency = pim_latency + cxl_latency + acc_latency
        token_latency = transformer_block_latency * TransformerBlock_number[args.model] + embedding_latency[channels_per_block] + InOut_latency
        throughput = 1000 / token_latency * pp
        
        energy_token = {}
        power_alldv = {}
        PCIE = embedding_size[args.model] * 10 + ffn_size[args.model] * 2 if args.model_parallel else embedding_size[args.model]
        energy_main, latency_main = power_calculator(stats, PCIE, n_heads[args.model], embedding_size[args.model], seqlen, gqa_factor[args.model])
        if args.model_parallel:
            pipeline_stages = args.num_devices // args.FC_devices
            FC_path = f"../trace/{args.num_channels}_channels_per_device/pipeline_parallel/{args.model}/trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt.log"
            stats_FC = command_processor(FC_path)
            energy_FC, latency_FC = power_calculator(stats_FC, PCIE, n_heads[args.model], embedding_size[args.model], seqlen, gqa_factor[args.model])
            for comp in energy_main.keys():
                energy_token[comp] = (energy_main[comp] + energy_FC[comp] * (args.FC_devices - 1)) * TransformerBlock_number[args.model]
                power_alldv[comp] = (energy_main[comp] + energy_FC[comp] * (args.FC_devices - 1)) * pipeline_stages / stats["latency"]
        else:
            for comp in energy_main.keys():
                energy_token[comp] = energy_main[comp] * utilized_devices
                power_alldv[comp] = energy_main[comp] * utilized_devices / stats["latency"]
        total_energy = 0
        for comp in energy_token.keys():
            total_energy += energy_token[comp]
        print(energy_token)
        print(total_energy)
        exit(0)
        total_power = 0
        for comp in power_alldv.keys():
            total_power += power_alldv[comp]
        device_utilization = 1.0 * utilized_devices / args.num_devices
                            
        new_result = {
            'Model': args.model,
            'Device number': args.num_devices,
            'Pipeline parallelism': pp,
            'Channels per device': args.num_channels,
            'Channels per block': channels_per_block,
            'Sequence length': seqlen,
            'PIM latency': pim_latency,
            'CXL latency': cxl_latency,
            'Acc latency': acc_latency,
            'TransformerBlock latency': transformer_block_latency,
            'Embedding latency': embedding_latency[channels_per_block],
            'Token latency': token_latency,
            'Throughput': throughput,
            'Token energy': total_energy,
            'Total power': total_power,
            'Device utilization': device_utilization
        }
        new_result_df = pd.DataFrame([new_result])
        results_df = pd.concat([results_df, new_result_df], ignore_index=True)

    # Save the DataFrame to a CSV file
    results_df.to_csv(args.simulation_result_path, index=False)
    # print(results_df)

def process_throughputs(args):

    if os.path.exists(args.simulation_result_path):
        df = pd.read_csv(args.simulation_result_path)
    else:
        raise ValueError(f"File {args.simulation_result_path} does not exist. Generate simulation results first.")

    if not args.model_parallel:

        if args.phase == "prefill":
            df = df[(df['Model'] == args.model) & (df['Sequence length'] < args.prefill)]
        elif args.phase == "decoding":
            df = df[(df['Model'] == args.model) & (df['Sequence length'] >= args.prefill)]
        else:
            df = df[(df['Model'] == args.model)]
        average_throughput = df['Throughput'].mean()
        average_energy = df['Token energy'].mean()

        if os.path.exists(args.final_result_path):
            results_df = pd.read_csv(args.final_result_path)
        else:
            columns = ['Model', 'Device number', 'Pipeline parallelism', 'Tensor parallelism', 'Phase', 'Throughput', 'Energy']
            results_df = pd.DataFrame(columns=columns)

        new_result = {
            'Model': args.model,
            'Device number': args.num_devices,
            'Pipeline parallelism': TransformerBlock_number[args.model],
            'Tensor parallelism': 1,
            'Phase': args.phase,
            'Throughput': average_throughput,
            'Energy': average_energy
        }
        new_result_df = pd.DataFrame([new_result])
        results_df = pd.concat([results_df, new_result_df], ignore_index=True)
    
    results_df.to_csv(args.final_result_path, index=False)


if args.generate_trace:
    generate_trace(args)
    
if args.simulate_trace:
    simulate_trace(args)

if args.process_results:
    process_results(args)
    
if args.update_csv:
    update_csv(args)
    
if args.process_throughputs:
    process_throughputs(args)
