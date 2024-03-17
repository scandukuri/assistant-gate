#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops-hgx-1
#SBATCH --gres=gpu:4  # Requesting four GPUs
#SBATCH --mem=428G 
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH --output=script-logs-3-mistral-ablation/m2-A-log-probs.out
#SBATCH --error=script-logs-3-mistral-ablation/m2-A-log-probs.err

# Load conda environment
source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
source activate assistant-gate

# navigate to python script parent directory
cd ~/research_projects/assistant-gate/experiments/star-3-mistral-ablation/log-probs/

python likelihood-qa-experimental.py model=m0 qa_model=m2 condition=qa-experimental split=A
