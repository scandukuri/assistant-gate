#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops-hgx-1
#SBATCH --gres=gpu:2  # Requesting four GPUs
#SBATCH --mem=256G 
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=m1-log-probs.out
#SBATCH --error=m1-log-probs.err

# Load conda environment
source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
source activate assistant-gate

# navigate to python script parent directory
cd ~/research_projects/assistant-gate/experiments/v5/log-probs/

# python likelihood-neg-control.py qa_model=m1-confusion2 condition=neg-control split=test
# python likelihood-pos-control.py qa_model=m1-confusion2 condition=pos-control split=test
python likelihood-qa-experimental.py qa_model=m1-confusion2 condition=qa-experimental split=test

sbatch /sailhome/andukuri/research_projects/assistant-gate/experiments/v5/log-probs/scripts/m1-confusion2-filter.sh