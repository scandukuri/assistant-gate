#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops2
#SBATCH --gres=gpu:2  # Requesting two GPUs
#SBATCH --mem=256G 
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=job_output.%j.out
#SBATCH --error=job_output.%j.err

# Load conda environment
source /scr/andukuri/miniconda3/etc/profile.d/conda.sh

source activate assistant-gate

# navigate to python script parent directory
cd ~/research_projects/assistant-gate/experiments/v2/conditions/pos-control

python determine-likelihood.py model=mistral-7b-instruct-v02-vllm