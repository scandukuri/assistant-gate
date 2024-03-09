#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops2
#SBATCH --gres=gpu:4  # Requesting four GPUs
#SBATCH --mem=512G 
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH --output=m0-test.out
#SBATCH --error=m0-test.err

# Load conda environment
source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
source activate assistant-gate

# navigate to python script parent directory
cd ~/research_projects/assistant-gate/experiments/v5/simulate-conversations/

python generate.py qa_model=m0 human_model=m0-hf split=test