## PIM is All You Need: A CXL-Enabled GPU-Free System for LLM Inference

* Figure 11, cost model python script
* Figure 12(a~c), simulation
* Figure 12(d), GPU raw data and simulation
* Figure 13, GPU raw data and simulation
* Figure 14(a, c), GPU raw data and simulation
* Figure 14(b), GPU raw data

### Prepare Environment
```bash
conda create -n cent_ae python=3.10
conda activate cent_ae
pip install pandas matplotlib ipykernel torch
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

### Run Simulation
```bash
cd cent_simulation
bash simulation.sh
bash parse_results.sh > throughput.txt
```

### Figure 2(a) 
Llama2-70B inference query latency increases with larger batches on 4 A100 80GB GPUs, with prompt=512, decoding=3584
```bash
python figure_2a/figure.py
```

### Figure 11
The CXL controller costs are broken down into die, packaging and Non Recurring Engineering (NRE) components. The die cost is derived from the wafer cost, considering the CXL controller die area and yield rate. The cost of 2D packaging is assumed to be 29% of chip cost (die and package). The NRE cost is
influenced by chip production volumes.
```bash
python figure_11.py
```