#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops2
#SBATCH --gres=gpu:0  # Requesting 0 GPUs
#SBATCH --mem=32G 
#SBATCH --cpus-per-task=20
#SBATCH --time=48:00:00
#SBATCH --output=script-logs-1-mistral-ablation/m0-A-filter.out
#SBATCH --error=script-logs-1-mistral-ablation/m0-A-filter.err

# Load conda environment
source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
source activate assistant-gate

# navigate to python script parent directory
cd ~/research_projects/assistant-gate/experiments/star-1-mistral-ablation/log-probs/

python filter.py model=m0 qa_model=m0 condition=qa-experimental split=A
