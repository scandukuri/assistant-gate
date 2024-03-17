#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops2
#SBATCH --gres=gpu:0  # Requesting four GPUs
#SBATCH --mem=36G 
#SBATCH --cpus-per-task=20
#SBATCH --time=48:00:00
#SBATCH --output=script-logs-3-mistral-ablation/m2-test-filter.out
#SBATCH --error=script-logs-3-mistral-ablation/m2-test-filter.err

# Load conda environment
source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
source activate assistant-gate

# navigate to python script parent directory
cd ~/research_projects/assistant-gate/experiments/star-3-mistral-ablation/log-probs/

# python filter.py qa_model=m2 condition=qa-experimental split=test
python filter-pos-control.py qa_model=m2 condition=pos-control-1 split=test
# python filter-pos-control.py qa_model=m2 condition=pos-control-2 split=test