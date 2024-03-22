#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops-hgx-1
#SBATCH --gres=gpu:0  # Requesting four GPUs
#SBATCH --mem=32G 
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output=script-logs-1-mistral-ablation/m3-test-log-probs-shuffled-filter.out
#SBATCH --error=script-logs-1-mistral-ablation/m3-test-log-probs-shuffled-filter.err

# Load conda environment
source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
source activate assistant-gate

# navigate to python script parent directory
cd ~/research_projects/assistant-gate/experiments/star-1-mistral-ablation/shuffled-log-probs/

python filter.py model=m0 qa_model=m3 condition=qa-experimental split=test