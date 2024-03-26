#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops-hgx-1
#SBATCH --gres=gpu:4  # Requesting four GPUs
#SBATCH --mem=468G 
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH --output=script-logs-2-gemma-ablation/m1-B.out
#SBATCH --error=script-logs-2-gemma-ablation/m1-B.err

# Load conda environment
source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
source activate assistant-gate

# navigate to python script parent directory
cd ~/research_projects/assistant-gate/experiments/star-2-gemma-ablation/simulate-conversations/

# Assuming n is the number of times you want to loop
# n MUST be the same as args.MAX_TURNS
n=3  # for example, replace 10 with the number of iterations you desire

for ((i=1; i<=n; i++))
do
    # Call generate-qa.py script
    python generate-qa.py qa_model=m1 human_model=gemma split=B turn="t$i"
    # Call generate-human.py script
    python generate-human.py qa_model=m1 human_model=gemma split=B turn="t$i"
done

