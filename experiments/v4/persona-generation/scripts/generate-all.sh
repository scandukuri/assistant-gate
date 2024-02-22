#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops-hgx-1
#SBATCH --gres=gpu:2  # Requesting four GPUs
#SBATCH --mem=256G 
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=generate-all-personas.out
#SBATCH --error=generate-all-personas.err

# Load conda environment
source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
source activate assistant-gate

# navigate to python script parent directory
cd /sailhome/andukuri/research_projects/assistant-gate/experiments/v4/persona-generation

python generate.py model=mixtral-8x7b-instruct-vllm split=A
python generate.py model=mixtral-8x7b-instruct-vllm split=B
python generate-test.py model=mixtral-8x7b-instruct-vllm split=test