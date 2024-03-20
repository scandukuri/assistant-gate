#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops2
#SBATCH --gres=gpu:2  # Requesting four GPUs
#SBATCH --mem=256G 
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=m0c-l.out
#SBATCH --error=m0c-l.err

# Load conda environment
source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
source activate assistant-gate

# navigate to python script parent directory
cd ~/research_projects/assistant-gate/experiments/v5/log-probs/

python likelihood-neg-control.py qa_model=m0-confusion condition=neg-control split=test
python likelihood-pos-control.py qa_model=m0-confusion condition=pos-control split=test
python likelihood-qa-experimental.py qa_model=m0-confusion condition=qa-experimental split=test

sbatch /sailhome/andukuri/research_projects/assistant-gate/experiments/v5/log-probs/scripts/m0-confusion-filter.sh

