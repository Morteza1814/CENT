import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─────────────── USER SETTINGS ─────────────── #
prefill_sizes       = [256, 512, 1024]          # try any list you like
requested_decodes   = [128, 512, 1024, 3584]    # original wish-list
context_len         = 4096                      # hard limit
transformer_blocks  = 80
src_csv             = "simulation_results.csv"
# out_data_csv        = "figure_source_data/figure_14d_cent_only.csv"
out_fig_pdf         = "figures/figure_14d_cent_only.pdf"
# ───────────────────────────────────────────── #

# CENT simulation results
df = pd.read_csv(src_csv)
df = df[
    (df["Model"] == "Llama2-70B") &
    (df["Device number"] == 32) &
    (df["Pipeline parallelism"] == transformer_blocks) &
    (df["Tensor parallelism"] == 1)
]

records, prefill_vals, decoding_vals, x_labels = [], [], [], []

for prefill in prefill_sizes:
    # average token latency for sequences ≤ prefill
    prefill_latency = (
        df[df["Sequence length"] <= prefill]["Token latency (ms)"].mean()
        * prefill / 1000 / 60
    )

    seen_decode_sizes = set()  # avoid duplicates per‐prefill

    for req_d in requested_decodes:
        # Adjust decoding size to honour context_len
        eff_d = min(req_d, max(context_len - prefill, 0))

        # Skip if no room or we already plotted this eff_d for this prefill
        if eff_d == 0 or eff_d in seen_decode_sizes:
            continue
        seen_decode_sizes.add(eff_d)

        dec_latency = (
            df[(df["Sequence length"] > prefill) &
               (df["Sequence length"] <= prefill + eff_d)]["Token latency (ms)"].mean()
            * eff_d / 1000 / 60
        )

        prefill_vals.append(prefill_latency)
        decoding_vals.append(dec_latency)
        x_labels.append(f"{prefill}_{eff_d}")

        records.append({
            "Prefill size": prefill,
            "Decoding tokens": eff_d,
            "Prefill latency (min)": prefill_latency,
            "Decoding latency (min)": dec_latency,
        })

# ───────────── save raw numbers ───────────── #
# os.makedirs("figure_source_data", exist_ok=True)
# pd.DataFrame(records).to_csv(out_data_csv, index=False)

# ─────────────────── PLOT ─────────────────── #
x = np.arange(len(x_labels))
prefill_arr  = np.array(prefill_vals)
decoding_arr = np.array(decoding_vals)

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(x, prefill_arr,  color="lightskyblue", edgecolor="black", label="Prefill")
ax.bar(x, decoding_arr, bottom=prefill_arr,
       color="sandybrown",  edgecolor="black", label="Decoding")

ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=9, rotation=90, ha="center")  # vertical labels
ax.set_ylabel("Query Latency (minutes)", fontsize=12)
ax.set_ylim(0, (prefill_arr + decoding_arr).max() * 1.1)
ax.legend(fontsize=10, loc="upper left")
ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig(out_fig_pdf)
plt.close()
