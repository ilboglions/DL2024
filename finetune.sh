#!/bin/bash
#SBATCH --job-name=mia
#SBATCH --gpus=rtx_4090:4
#SBATCH --mem-per-cpu=4G
#SBATCH --ntasks=1 --cpus-per-task=16
#SBATCH --time=100:00:00
export CUDA_VISIBLE_DEVICES=0,1,2,3

deepspeed --num_gpus=4 src/train.py --config configs/opt.toml
