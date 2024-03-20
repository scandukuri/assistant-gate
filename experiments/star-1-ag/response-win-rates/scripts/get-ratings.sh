#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops-hgx-1
#SBATCH --gres=gpu:0  # Requesting four GPUs
#SBATCH --mem=128G 
#SBATCH --cpus-per-task=24
#SBATCH --time=48:00:00
#SBATCH --output=script-logs/get-ratings.out
#SBATCH --error=script-logs/get-ratings.err

# Load conda environment
source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
source activate assistant-gate

# navigate to python script parent directory
cd ~/research_projects/assistant-gate/experiments/star-1-ag/response-win-rates/

python get-ratings.py qa_model=m0 qa_model_2=m1 split=test tokenizer=m0
python get-ratings.py qa_model=m0 qa_model_2=m2 split=test tokenizer=m0
