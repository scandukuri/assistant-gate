#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops-hgx-1
#SBATCH --gres=gpu:1  # Requesting one GPUs
#SBATCH --mem=64G 
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=m1-shuffle.out
#SBATCH --error=m1-shuffle.err

# Load conda environment
source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
source activate assistant-gate

# navigate to python script parent directory
cd ~/research_projects/assistant-gate/experiments/v9/simulate-conversations/

python shuffle.py qa_model=m0 split=test