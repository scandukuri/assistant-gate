#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops2
#SBATCH --gres=gpu:0  # Requesting one GPUs
#SBATCH --mem=64G 
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=script-logs-3-mistral-ablation/m2-test-pool.out
#SBATCH --error=script-logs-3-mistral-ablation/m2-test-pool.err

# Load conda environment
source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
source activate assistant-gate

# navigate to python script parent directory
cd ~/research_projects/assistant-gate/experiments/star-3-mistral-ablation/simulate-conversations/

python pool-conversations.py qa_model=m2 split=test
