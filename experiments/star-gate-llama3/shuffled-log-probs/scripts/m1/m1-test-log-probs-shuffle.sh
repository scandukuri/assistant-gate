#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops-hgx-1
#SBATCH --gres=gpu:4  # Requesting four GPUs
#SBATCH --mem=360G 
#SBATCH --cpus-per-task=54
#SBATCH --time=48:00:00
#SBATCH --output=script-logs-llama3/m1-test-log-probs-shuffled.out
#SBATCH --error=script-logs-llama3/m1-test-log-probs-shuffled.err

# Load conda environment
source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
source activate assistant-gate

# navigate to python script parent directory
cd ~/research_projects/assistant-gate/experiments/star-gate-llama3/shuffled-log-probs/

python likelihood-qa-experimental.py model=m0 qa_model=m1 condition=qa-experimental split=test