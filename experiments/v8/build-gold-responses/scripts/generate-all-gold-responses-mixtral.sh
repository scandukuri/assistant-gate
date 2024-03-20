#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops2
#SBATCH --gres=gpu:2  # Requesting four GPUs
#SBATCH --mem=256G 
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=script-logs/gold-mixtral.out
#SBATCH --error=script-logs/gold-mixtral.err

# Load conda environment
source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
source activate assistant-gate

# navigate to python script parent directory
cd /sailhome/andukuri/research_projects/assistant-gate/experiments/v8/build-gold-responses

python generate-mixtral.py split=test model=mixtral-8x7b-instruct-vllm
python generate-mixtral.py split=A model=mixtral-8x7b-instruct-vllm
python generate-mixtral.py split=B model=mixtral-8x7b-instruct-vllm
