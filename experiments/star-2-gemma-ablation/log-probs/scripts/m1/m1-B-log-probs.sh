#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops-hgx-1
#SBATCH --gres=gpu:2  # Requesting four GPUs
#SBATCH --mem=128G 
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=script-logs-2-gemma-ablation/m1-B-log-probs.out
#SBATCH --error=script-logs-2-gemma-ablation/m1-B-log-probs.err

# Load conda environment
source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
source activate assistant-gate

# navigate to python script parent directory
cd ~/research_projects/assistant-gate/experiments/star-2-gemma-ablation/log-probs/

python likelihood-qa-experimental.py model=m0 qa_model=m1 condition=qa-experimental split=B
