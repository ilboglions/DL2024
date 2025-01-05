#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=a100-pcie-40gb:1
#SBATCH --time=48:00:00
#SBATCH --job-name="olmo7b-tt"
#SBATCH --mem-per-cpu=4096
#SBATCH --output="results/time_travel/OLMo_7b/logs.txt"

python src/eval_contamination.py --config results/config_contamination.toml
