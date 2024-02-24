#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops-hgx-1
#SBATCH --gres=gpu:0  # Requesting four GPUs
#SBATCH --mem=64G 
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=m1-log-probs.out
#SBATCH --error=m1-log-probs.err

# Load conda environment
source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
source activate assistant-gate

# navigate to python script parent directory
cd ~/research_projects/assistant-gate/experiments/v5/log-probs/

python filter.py qa_model=m0-confusion2 condition=qa-experimental split=test
python filter-pos-control.py qa_model=m0-confusion2 condition=pos-control split=test