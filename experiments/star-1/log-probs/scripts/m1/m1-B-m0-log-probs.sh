#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops-hgx-1
#SBATCH --gres=gpu:2  # Requesting four GPUs
#SBATCH --mem=256G 
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=~/script-logs/m1-B-log-probs.out
#SBATCH --error=~/script-logs/m1-B-log-probs.err

# Load conda environment
source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
source activate assistant-gate

# navigate to python script parent directory
cd ~/research_projects/assistant-gate/experiments/star-1/log-probs/

# python model-t-log-probs/likelihood-neg-control.py qa_model=m1 condition=neg-control split=B
# python model-t-log-probs/likelihood-pos-control-1.py qa_model=m1 condition=pos-control-1 split=B
# python model-t-log-probs/likelihood-pos-control-2.py qa_model=m1 condition=pos-control-2 split=B
python model-t-log-probs/likelihood-qa-experimental.py qa_model=m1 condition=qa-experimental split=B