import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─────────────── USER SETTINGS ─────────────── #
prefill_sizes       = [32, 64, 128, 256, 512, 1024, 2048, 3584, 3840, 3968, 4032, 4064]
context_len         = 4096             
tensor_parallelism = 1
pipeline_parallelism = 8
src_csv             = "simulation_results_7B_32sqgap.csv"
# out_data_csv        = "figure_source_data/figure_14d_cent_only.csv"
out_data_csv         = f"7B_d8_tp{tensor_parallelism}_pp{pipeline_parallelism}_latency.csv"
# ───────────────────────────────────────────── #

# CENT simulation results
df = pd.read_csv(src_csv)
df = df[
    (df["Model"] == "Llama2-7B") &
    (df["Device number"] == 8) &
    (df["Pipeline parallelism"] == pipeline_parallelism) &
    (df["Tensor parallelism"] == tensor_parallelism)
]

records, prefill_vals, decoding_vals, x_labels = [], [], [], []

for prefill in prefill_sizes:
    max_decode = max(context_len - prefill, 0)
    decode_sizes = []

    # Start with 32 and multiply by 2 until >= max_decode
    d = 32
    while d < max_decode:
        decode_sizes.append(d)
        d *= 2
    if max_decode > 0:
        decode_sizes.append(max_decode)  # include final decode (possibly not power of two)

    decode_sizes = list(dict.fromkeys(decode_sizes))  # remove duplicates

    # average token latency for sequences ≤ prefill
    prefill_latency = (
        df[df["Sequence length"] <= prefill]["Token latency (ms)"].mean()
        * prefill / 1000 / 60
    )

    prefill_latency_ms  = (
        df[df["Sequence length"] <= prefill]["Token latency (ms)"].mean()
        * prefill
    )

    for dec in decode_sizes:
        dec_latency = (
            df[(df["Sequence length"] > prefill) &
               (df["Sequence length"] <= prefill + dec)]["Token latency (ms)"].mean()
            * dec / 1000 / 60
        )

        dec_latency_ms = (
            df[(df["Sequence length"] > prefill) &
               (df["Sequence length"] <= prefill + dec)]["Token latency (ms)"].mean()
            * dec
        )

        prefill_vals.append(prefill_latency)
        decoding_vals.append(dec_latency)
        x_labels.append(f"{prefill}_{dec}")

        records.append({
            "model": "Llama2-7B",
            "system": "CENT",
            "batch": pipeline_parallelism,
            "input": prefill,
            "output": dec,
            "Total tokens": prefill + dec,
            "Prefill latency (min)": prefill_latency_ms,
            "Decoding latency (min)": dec_latency_ms,
        })


# ───────────── save raw numbers ───────────── #
# Derive “figure_14d_cent_only.csv” from “figure_14d_cent_only.pdf”
# out_data_csv = os.path.splitext(out_fig_pdf)[0] + ".csv"
csv_dir      = os.path.dirname(out_data_csv)
# only create a directory if one is actually present in the path
if csv_dir:                       # '' means “current directory” → nothing to create
    os.makedirs(csv_dir, exist_ok=True)

pd.DataFrame(records).to_csv(out_data_csv, index=False)
# ─────────────────── PLOT ─────────────────── #
# x = np.arange(len(x_labels))
# prefill_arr  = np.array(prefill_vals)
# decoding_arr = np.array(decoding_vals)

# fig, ax = plt.subplots(figsize=(8, 4))
# ax.bar(x, prefill_arr,  color="lightskyblue", edgecolor="black", label="Prefill")
# ax.bar(x, decoding_arr, bottom=prefill_arr,
#        color="sandybrown",  edgecolor="black", label="Decoding")

# ax.set_xticks(x)
# ax.set_xticklabels(x_labels, fontsize=9, rotation=90, ha="center")  # vertical labels
# ax.set_ylabel("Query Latency (minutes)", fontsize=12)
# ax.set_ylim(0, (prefill_arr + decoding_arr).max() * 1.1)
# ax.legend(fontsize=10, loc="upper left")
# ax.grid(axis="y", linestyle="--", alpha=0.5)

# plt.tight_layout()
# os.makedirs("figures", exist_ok=True)
# plt.savefig(out_fig_pdf)
# plt.close()
