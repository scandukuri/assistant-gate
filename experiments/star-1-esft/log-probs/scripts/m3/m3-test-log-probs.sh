#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops2
#SBATCH --gres=gpu:4  # Requesting four GPUs
#SBATCH --mem=400G 
#SBATCH --cpus-per-task=80
#SBATCH --time=48:00:00
#SBATCH --output=script-logs-1-esft/m3-test-log-probs.out
#SBATCH --error=script-logs-1-esft/m3-test-log-probs.err

# Load conda environment
source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
source activate assistant-gate

# navigate to python script parent directory
cd ~/research_projects/assistant-gate/experiments/star-1-esft/log-probs/

python likelihood-neg-control.py model=m0 qa_model=m3 condition=neg-control split=test
python likelihood-pos-control-1.py model=m0 qa_model=m3 condition=pos-control-1 split=test
python likelihood-pos-control-2.py model=m0 qa_model=m3 condition=pos-control-2 split=test
python likelihood-qa-experimental.py model=m0 qa_model=m3 condition=qa-experimental split=test
