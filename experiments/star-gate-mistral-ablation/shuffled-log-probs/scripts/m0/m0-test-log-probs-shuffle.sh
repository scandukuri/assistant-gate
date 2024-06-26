#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops-hgx-1
#SBATCH --gres=gpu:4  # Requesting four GPUs
#SBATCH --mem=400G 
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH --output=script-logs-2-mistral-ablation/m0-test-log-probs-shuffled.out
#SBATCH --error=script-logs-2-mistral-ablation/m0-test-log-probs-shuffled.err

# Load conda environment
source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
source activate assistant-gate

# navigate to python script parent directory
cd ~/research_projects/assistant-gate/experiments/star-2-mistral-ablation/shuffled-log-probs/

python likelihood-qa-experimental.py model=m0 qa_model=m0 condition=qa-experimental split=test