#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops2
#SBATCH --gres=gpu:0  # Requesting four GPUs
#SBATCH --mem=32G 
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=script-logs-2-bsft/gold.out
#SBATCH --error=script-logs-2-bsft/gold.err

# Load conda environment
source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
source activate assistant-gate

# navigate to python script parent directory
cd /sailhome/andukuri/research_projects/assistant-gate/experiments/star-2-bsft/build-gold-responses

python generate.py split=A model=gpt4
python generate.py split=B model=gpt4