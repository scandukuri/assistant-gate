#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops-hgx-1
#SBATCH --gres=gpu:0  # Requesting no GPUs
#SBATCH --mem=1G 
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output=script-logs-2-bsft/extract-prompts.out
#SBATCH --error=script-logs-2-bsft/extract-prompts.err

# Load conda environment
source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
source activate assistant-gate

# navigate to python script parent directory
cd /sailhome/andukuri/research_projects/assistant-gate/experiments/star-2-bsft/instruct-questions

python extract-questions.py