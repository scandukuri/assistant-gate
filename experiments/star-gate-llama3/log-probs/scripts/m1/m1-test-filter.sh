#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops-hgx-1
#SBATCH --gres=gpu:0  # Requesting four GPUs
#SBATCH --mem=36G 
#SBATCH --cpus-per-task=20
#SBATCH --time=48:00:00
#SBATCH --output=script-logs-llama3/m1-test-filter.out
#SBATCH --error=script-logs-llama3/m1-test-filter.err

# Load conda environment
source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
source activate assistant-gate

# navigate to python script parent directory
cd ~/research_projects/assistant-gate/experiments/star-gate-llama3/log-probs/

python filter.py qa_model=m1 condition=qa-experimental split=test
python filter-pos-control.py qa_model=m1 condition=pos-control-1 split=test
python filter-pos-control.py qa_model=m1 condition=pos-control-2 split=test