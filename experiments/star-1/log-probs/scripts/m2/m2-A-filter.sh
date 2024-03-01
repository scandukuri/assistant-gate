#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops-hgx-1
#SBATCH --gres=gpu:0  # Requesting four GPUs
#SBATCH --mem=96G 
#SBATCH --cpus-per-task=20
#SBATCH --time=48:00:00
#SBATCH --output=script-logs/m2-A-filter.out
#SBATCH --error=script-logs/m2-A-filter.err

# Load conda environment
source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
source activate assistant-gate

# navigate to python script parent directory
cd ~/research_projects/assistant-gate/experiments/star-1/log-probs/

python model-t-log-probs/filter.py qa_model=m2 condition=qa-experimental split=A
python model-t-log-probs/filter-pos-control.py qa_model=m2 condition=pos-control-1 split=A
python model-t-log-probs/filter-pos-control.py qa_model=m2 condition=pos-control-2 split=A