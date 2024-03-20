#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops-hgx-1
#SBATCH --gres=gpu:2  # Requesting four GPUs
#SBATCH --mem=128G 
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=script-logs/m0-A.out
#SBATCH --error=script-logs/m0-A.err

# Load conda environment
source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
source activate assistant-gate

# navigate to python script parent directory
cd ~/research_projects/assistant-gate/experiments/v9/simulate-conversations/

# Assuming n is the number of times you want to loop
# n MUST be the same as args.MAX_TURNS
n=3  # for example, replace 10 with the number of iterations you desire


# Call generate-qa.py script
python generate-qa.py qa_model=m0 human_model=M0 split=A
    


