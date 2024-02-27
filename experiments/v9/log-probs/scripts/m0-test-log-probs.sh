#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops-hgx-1
#SBATCH --gres=gpu:2  # Requesting two GPUs
#SBATCH --mem=256G 
#SBATCH --cpus-per-task=20
#SBATCH --time=48:00:00
#SBATCH --output=m0-lt.out
#SBATCH --error=m0-lt.err

# Load conda environment
source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
source activate assistant-gate

# navigate to python script parent directory
cd ~/research_projects/assistant-gate/experiments/v9/log-probs/

python likelihood-neg-control.py qa_model=m0 condition=neg-control split=test
python likelihood-pos-control-1.py qa_model=m0 condition=pos-control-1 split=test
python likelihood-pos-control-2.py qa_model=m0 condition=pos-control-2 split=test
python likelihood-qa-experimental.py qa_model=m0 condition=qa-experimental split=test