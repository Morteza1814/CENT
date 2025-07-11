#!/bin/bash

#SBATCH --ntasks=1            
#SBATCH --cpus-per-task=64    
#SBATCH --job-name=run_sim
#SBATCH --partition=cpu
#SBATCH --mem=100000
#SBATCH --time=2-00:00:00
#SBATCH --output=run_sim.txt
#SBATCH --error=run_sim.txt

bash simulation.sh 64 128