#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops-hgx-1
#SBATCH --gres=gpu:2  # Requesting four GPUs
#SBATCH --mem=256G 
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=m0-l.out
#SBATCH --error=m0-l.err

# Load conda environment
source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
source activate assistant-gate

# navigate to python script parent directory
cd ~/research_projects/assistant-gate/experiments/v8/log-probs/

python likelihood-neg-control.py qa_model=m0 condition=neg-control split=A
python likelihood-pos-control.py qa_model=m0 condition=pos-control split=A
python likelihood-qa-experimental.py qa_model=m0 condition=qa-experimental split=A
python likelihood-neg-control.py qa_model=m0 condition=neg-control split=test
python likelihood-pos-control.py qa_model=m0 condition=pos-control split=test
python likelihood-qa-experimental.py qa_model=m0 condition=qa-experimental split=test