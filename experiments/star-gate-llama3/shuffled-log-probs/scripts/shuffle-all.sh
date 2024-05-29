#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops-hgx-1
#SBATCH --gres=gpu:0  # Requesting 0 GPUs
#SBATCH --mem=32G 
#SBATCH --cpus-per-task=20
#SBATCH --time=48:00:00
#SBATCH --output=script-logs-llama3/shuffle.out
#SBATCH --error=script-logs-llama3/shuffle.err

# Load conda environment
source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
source activate assistant-gate

# navigate to python script parent directory
cd ~/research_projects/assistant-gate/experiments/star-gate-llama3/shuffled-log-probs/

python shuffle.py qa_model=m0 split=test
python shuffle.py qa_model=m1 split=test
python shuffle.py qa_model=m2 split=test
python shuffle.py qa_model=m3 split=test
