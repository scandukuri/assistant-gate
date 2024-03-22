#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops-hgx-1
#SBATCH --gres=gpu:4  # Requesting four GPUs
#SBATCH --mem=400G 
#SBATCH --cpus-per-task=48
#SBATCH --time=48:00:00
#SBATCH --output=script-logs-1-mistral-ablation/get-responses-randomized.out
#SBATCH --error=script-logs-1-mistral-ablation/get-responses-randomized.err

# Load conda environment
source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
source activate assistant-gate

# navigate to python script parent directory
cd ~/research_projects/assistant-gate/experiments/star-1-mistral-ablation/response-win-rates-randomized/

python get-responses.py answer_model=baseline qa_model=baseline split=test
#python get-responses.py answer_model=m0 qa_model=m0 split=test
# python get-responses.py answer_model=m1 qa_model=m1 split=test
# python get-responses.py answer_model=m2 qa_model=m2 split=test
# python get-responses.py answer_model=m3 qa_model=m3 split=test