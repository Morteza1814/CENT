# Artifcat of the CENT Paper, ASPLOS 2025

This repository provides the following artifact required for the evaluation of the **CENT**, "PIM is All You Need: A CXL-Enabled GPU-Free System for LLM Inference" paper published in ASPLOS 2025:

## Directory Structure

TODO

## Dependencies

AiM Simulator is tested and verified with `g++-12` and `clang++-15`.

## Build

Prepare the environment using the following commands:

```bash
conda create -n cent_ae python=3.10
conda activate cent_ae
pip install -r requirements.txt

git submodule add https://github.com/arkhadem/ramulator2.git ramulator2
cd ramulator2
# only gcc-12 is supported
mkdir build
cd build
cmake ..
make -j
cp ./ramulator2 ../ramulator2
cd ..
```

## Artifact Scripts

Remove old results
```bash
rm simulation_results.csv
rm processed_results.csv
rm figures/*
rm figure_source_data/*
```

### Run Simulation
```bash
cd cent_simulation
bash simulation.sh
bash process_results.sh
```

### Figure 11
The CXL controller costs are broken down into die, packaging and Non Recurring Engineering (NRE) components. The die cost is derived from the wafer cost, considering the CXL controller die area and yield rate. The cost of 2D packaging is assumed to be 29% of chip cost (die and package). The NRE cost is
influenced by chip production volumes.
```bash
python figure_11.py
```

### Figure 12
Analysis on Llama2-70B. (a) CENT achieves higher decoding throughputs with long context windows and 3.5K decoding sizes. (b) QoS analysis: CENT shows less query latency when achieving the similar to GPUs. (c) CENT latency breakdown with different parallelism strategies. (d) Prefill (In) and decoding (Out) latency comparison with different In/Out sizes, at maximum supported batches for both GPU and CENT.
```bash
python figure_12a.py
python figure_12b.py
python figure_12c.py
python figure_12d.py
```

### Figure 13
CENT speedup over GPU baselines. (a) Batch = 1 Latency comparison. (b) Throughput comparison under the highest batch size that CENT and GPU and achieve. (c) TCO normalized throughput comprison.
```bash
python figure_13a.py
python figure_13b.py
python figure_13c.py
```

### Figure 14
Power and energy analysis. (a) Power consumption, (b) GPU SM frequency and board power, and (c) energy efficiency of CENT and GPU for different stages of Llama2 models using the maximum batch size, 512 prefill tokens and 3584 decoding tokens.
```bash
python figure_14a.py
python figure_14c.py
```

## Citation

If you use *CENT*, please cite this paper:

> Yufeng Gu, Alireza Khadem, Sumanth Umesh, Ning Liang, Xavier Servot, Onur Mutlu, Ravi Iyer, and Reetuparna Das.
> *PIM is All You Need: A CXL-Enabled GPU-Free System for LLM Inference*,
> In 2025 International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS)

```
@inproceedings{cent,
  title={PIM is All You Need: A CXL-Enabled GPU-Free System for LLM Inference},
  author={Gu, Yufeng and Khadem, Alireza and Umesh, Sumanth, and Liang, Ning and Servot, Xavier and Mutlu, Onur and Iyer, Ravi and and Das, Reetuparna},
  booktitle={2025 International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS)}, 
  year={2025}
}
```

## Issues and bug reporting

We appreciate any feedback and suggestions from the community.
Feel free to raise an issue or submit a pull request on Github.
For assistance in using CENT, please contact: Yufeng Gu (yufenggu@umich.edu) and Alireza Khadem (arkhadem@umich.edu)

## Licensing

This repository is available under a [MIT license](/LICENSE).

## Acknowledgement

This work was supported in part by the NSF under the CAREER-1652294 and NSF-1908601 awards and by Intel gift.