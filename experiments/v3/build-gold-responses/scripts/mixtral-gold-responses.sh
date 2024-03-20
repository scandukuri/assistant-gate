#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops-hgx-1
#SBATCH --gres=gpu:2  # Requesting four GPUs
#SBATCH --mem=256G 
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=mixtral-gold.out
#SBATCH --error=mixtral-gold.err

# Load conda environment
source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
source activate assistant-gate

# install necessary env packages
cd ~/research_projects/assistant-gate


# navigate to python script parent directory
cd /sailhome/andukuri/research_projects/assistant-gate/experiments/v3/build-gold-responses

python generate.py model=mixtral-8x7b-instruct-vllm