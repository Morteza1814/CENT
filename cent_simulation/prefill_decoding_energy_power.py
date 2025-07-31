import os
import pandas as pd

# ─────────────── USER SETTINGS ─────────────── #
prefill_sizes       = [32, 64, 128, 256, 512, 1024, 2048, 3584, 3840, 3968, 4032, 4064]
context_len         = 4096
tensor_parallelism = 1
pipeline_parallelism = 8
src_csv             = "simulation_results_7B_32sqgap.csv"
out_csv             = f"7B_d8_tp{tensor_parallelism}_pp{pipeline_parallelism}_total_energy_with_power.csv"
# ───────────────────────────────────────────── #

# Column names (as in your file)
COL_SEQLEN = "Sequence length"
COL_EPT_mJ = "Token energy (mJ)"   # mJ/token
COL_PWR_W  = "Total power (W)"     # Watts

# Load & filter
df = pd.read_csv(src_csv)
df = df[
    (df["Model"] == "Llama2-7B") &
    (df["Device number"] == 8) &
    (df["Pipeline parallelism"] == pipeline_parallelism) &
    (df["Tensor parallelism"] == tensor_parallelism)
].copy()

# Required columns
for col in [COL_SEQLEN, COL_EPT_mJ, COL_PWR_W]:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

records = []

for prefill in prefill_sizes:
    max_decode = max(context_len - prefill, 0)

    # Decode buckets: 32, 128, 512, 2048, ... plus remainder
    decode_sizes, d = [], 32
    while d < max_decode:
        decode_sizes.append(d)
        d *= 2
    if max_decode > 0:
        decode_sizes.append(max_decode)
    decode_sizes = list(dict.fromkeys(decode_sizes))  # de-dup keep order

    pre_slice = df[df[COL_SEQLEN] <= prefill]
    if len(pre_slice) == 0:
        continue

    # Convert mJ/token → J/token
    avg_ept_prefill_Jpt = pre_slice[COL_EPT_mJ].mean() * 1e-3

    for dec in decode_sizes:
        dec_slice = df[(df[COL_SEQLEN] > prefill) & (df[COL_SEQLEN] <= prefill + dec)]
        if len(dec_slice) == 0:
            continue

        avg_ept_decode_Jpt = dec_slice[COL_EPT_mJ].mean() * 1e-3

        total_tokens = prefill + dec
        # Total energy across whole request (prefill + decode), in Joules
        E_total_J = avg_ept_prefill_Jpt * prefill + avg_ept_decode_Jpt * dec

        # One overall average power over the combined slice (prefill+decode)
        union_slice = df[df[COL_SEQLEN] <= prefill + dec]
        avg_total_power_W = union_slice[COL_PWR_W].mean()

        records.append({
            "model": "Llama2-7B",
            "system": "CENT",
            "batch": pipeline_parallelism,
            "input": prefill,
            "output": dec,
            "Total tokens": total_tokens,
            "Avg EPT prefill (J/token)": avg_ept_prefill_Jpt,
            "Avg EPT decode (J/token)": avg_ept_decode_Jpt,
            "Total energy (J)": E_total_J,
            "Avg total power (W)": avg_total_power_W,
        })

# Save CSV
csv_dir = os.path.dirname(out_csv)
if csv_dir:
    os.makedirs(csv_dir, exist_ok=True)
pd.DataFrame(records).to_csv(out_csv, index=False)
print(f"Saved: {out_csv}")
