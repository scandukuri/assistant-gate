#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops2
#SBATCH --gres=gpu:2  # Requesting four GPUs
#SBATCH --mem=256G 
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=m0-f.out
#SBATCH --error=m0-f.err

# Load conda environment
source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
source activate assistant-gate

# navigate to python script parent directory
cd ~/research_projects/assistant-gate/experiments/v8/log-probs/

python filter.py qa_model=m0 condition=qa-experimental split=A
python filter-pos-control.py qa_model=m0 condition=pos-control split=A
python filter.py qa_model=m0 condition=qa-experimental split=test
python filter-pos-control.py qa_model=m0 condition=pos-control split=test