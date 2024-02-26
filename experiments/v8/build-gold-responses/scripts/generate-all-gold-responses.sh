#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops-hgx-1
#SBATCH --gres=gpu:0  # Requesting four GPUs
#SBATCH --mem=16G 
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=gold.out
#SBATCH --error=gold.err

# Load conda environment
source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
source activate assistant-gate

# navigate to python script parent directory
cd /sailhome/andukuri/research_projects/assistant-gate/experiments/v8/build-gold-responses

#python generate.py split=A model=gpt4
#python generate.py split=B model=gpt4
#python generate.py split=test model=gpt4