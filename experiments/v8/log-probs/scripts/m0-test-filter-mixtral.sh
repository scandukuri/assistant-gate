#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops-hgx-1
#SBATCH --gres=gpu:0  # Requesting four GPUs
#SBATCH --mem=8G 
#SBATCH --cpus-per-task=20
#SBATCH --time=48:00:00
#SBATCH --output=m0-f.out
#SBATCH --error=m0-f.err

# Load conda environment
source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
source activate assistant-gate

# navigate to python script parent directory
cd ~/research_projects/assistant-gate/experiments/v8/log-probs/

python filter.py qa_model=m0 condition=qa-experimental split=test-mixtral
python filter-pos-control.py qa_model=m0 condition=pos-control-1 split=test-mixtral
python filter-pos-control.py qa_model=m0 condition=pos-control-2 split=test-mixtral