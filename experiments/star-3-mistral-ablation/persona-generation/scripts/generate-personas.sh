#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops2
#SBATCH --gres=gpu:0  # Requesting four GPUs
#SBATCH --mem=4G 
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=script-logs-3-mistral-ablation/generate-personas.out
#SBATCH --error=script-logs-3-mistral-ablation/generate-personas.err

# Load conda environment
source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
source activate assistant-gate

# navigate to python script parent directory
cd /sailhome/andukuri/research_projects/assistant-gate/experiments/star-3-mistral-ablation/persona-generation

python generate-personas-test.py model=gpt4 split=test
python generate-personas.py model=gpt4 split=A
python generate-personas.py model=gpt4 split=B