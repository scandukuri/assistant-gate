#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops2
#SBATCH --gres=gpu:0  # Requesting no GPUs
#SBATCH --mem=8G 
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output=script-logs-3-mistral-ablation/m1-B-pool.out
#SBATCH --error=script-logs-3-mistral-ablation/m1-B-pool.err

# Load conda environment
source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
source activate assistant-gate

# navigate to python script parent directory
cd ~/research_projects/assistant-gate/experiments/star-3-mistral-ablation/simulate-conversations/

python pool-conversations.py qa_model=m1 split=B
