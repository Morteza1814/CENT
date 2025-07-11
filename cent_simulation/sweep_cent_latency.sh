#!/bin/bash
#
#SBATCH --job-name=sweep_cent_latency
#SBATCH --partition=cpu          # change to gpu if your simulator needs GPUs
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=sweep_cent_latency.out
#SBATCH --error=sweep_cent_latency.out

# ───── 1. Sweep definitions ────────────────────────────────────────────────
prefill_list=(32 64 128 256 512 1024 2048 4096 8192 16384 32768)     # prompt sizes you want to test
decode_list=(32 64 128 256 512 1024 2048 4096 8192 16384 32768)         # max generation lengths to test

# Optional: start with a clean CSV
CSV_PATH="simulation_results_2.csv"
rm -f "$CSV_PATH"

# ─── 2.  Activate your Python environment ────────────────────────────────
# module load python/3.10

# ─── 3.  Run every (prefill, decode) combination sequentially ────────────
for PREFILL in "${prefill_list[@]}"; do
  for DECODE in "${decode_list[@]}"; do
    echo "▶ Running prefill=${PREFILL}, decode=${DECODE}"
    python3 run_sim.py \
        --prefill $PREFILL \
        --decoding $DECODE \
        --generate_trace \
        --simulate_trace \
        --process_results \
        --update_csv \
        --simulation_result_path "$CSV_PATH" \
        --model Llama2-70B \
        --num_devices 32
  done
done

echo "✓ Sweep finished. Consolidated results in $CSV_PATH"