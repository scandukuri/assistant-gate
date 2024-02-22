#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops-hgx-1
#SBATCH --gres=gpu:4  # Requesting four GPUs
#SBATCH --mem=364G 
#SBATCH --cpus-per-task=48
#SBATCH --time=48:00:00
#SBATCH --output=m1-test.out
#SBATCH --error=m1-test.err

# Load conda environment
source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
source activate assistant-gate

# navigate to python script parent directory
cd ~/research_projects/assistant-gate/experiments/v5/simulate-conversations/

python generate.py qa_model=m1 human_model=m0-hf split=test