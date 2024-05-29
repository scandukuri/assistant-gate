#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops-hgx-1
#SBATCH --gres=gpu:0  # Requesting 0 GPUs
#SBATCH --mem=16G 
#SBATCH --cpus-per-task=10
#SBATCH --time=48:00:00
#SBATCH --output=script-logs-llama3/m2-A-filter.out
#SBATCH --error=script-logs-llama3/m2-A-filter.err

# Load conda environment
source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
source activate assistant-gate

# navigate to python script parent directory
cd ~/research_projects/assistant-gate/experiments/star-gate-llama3/log-probs/

python filter.py model=m0 qa_model=m2 condition=qa-experimental split=A
