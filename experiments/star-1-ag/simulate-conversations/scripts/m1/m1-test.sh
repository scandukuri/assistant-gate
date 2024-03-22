#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops-hgx-1
#SBATCH --gres=gpu:2  # Requesting four GPUs
#SBATCH --mem=256G 
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=script-logs/m1-test.out
#SBATCH --error=script-logs/m1-test.err

# Load conda environment
source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
source activate assistant-gate

# navigate to python script parent directory
cd ~/research_projects/assistant-gate/experiments/star-1-ag/simulate-conversations/

# Assuming n is the number of times you want to loop
n=3  # for example, replace n with the number of iterations you desire


for ((i=1; i<=n; i++))
do
    # Call generate-qa.py script
    python generate-qa.py qa_model=m1 human_model=M0 split=test turn="t$i"
    # Call generate-human.py script
    python generate-human.py qa_model=m1 human_model=M0 split=test turn="t$i"
done