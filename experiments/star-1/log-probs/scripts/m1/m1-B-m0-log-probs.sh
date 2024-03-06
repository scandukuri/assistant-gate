#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops-hgx-1
#SBATCH --gres=gpu:4  # Requesting four GPUs
#SBATCH --mem=512G 
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH --output=script-logs/m1-B-m0-log-probs.out
#SBATCH --error=script-logs/m1-B-m0-log-probs.err

# Load conda environment
source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
source activate assistant-gate

# navigate to python script parent directory
cd ~/research_projects/assistant-gate/experiments/star-1/log-probs/

python model-0-log-probs/likelihood-qa-experimental-m0-lp.py model=m0 qa_model=m1 condition=qa-experimental split=B
