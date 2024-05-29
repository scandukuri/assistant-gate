#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops-hgx-1
#SBATCH --gres=gpu:8  # Requesting four GPUs
#SBATCH --mem=968G 
#SBATCH --cpus-per-task=96
#SBATCH --time=48:00:00
#SBATCH --output=script-logs-llama3/m0-A-log-probs.out
#SBATCH --error=script-logs-llama3/m0-A-log-probs.err

# Load conda environment
source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
source activate assistant-gate

# navigate to python script parent directory
cd ~/research_projects/assistant-gate/experiments/star-gate-llama3/log-probs/

python likelihood-qa-experimental.py model=m0 qa_model=m0 condition=qa-experimental split=A
