#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops2
#SBATCH --gres=gpu:0  # Requesting 0 GPUs
#SBATCH --mem=4G 
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=script-logs-1-mistral-ablation/gold-test.out
#SBATCH --error=script-logs-1-mistral-ablation/gold-test.err

# Load conda environment
source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
source activate assistant-gate

# navigate to python script parent directory
cd /sailhome/andukuri/research_projects/assistant-gate/experiments/star-1-mistral-ablation/build-gold-responses

python generate.py split=test model=gpt4