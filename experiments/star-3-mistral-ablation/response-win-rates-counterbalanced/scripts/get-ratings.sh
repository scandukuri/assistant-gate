#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops-hgx-1
#SBATCH --gres=gpu:0  # Requesting four GPUs
#SBATCH --mem=32G 
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=script-logs-3-mistral-ablation/get-ratings-counterbalanced.out
#SBATCH --error=script-logs-3-mistral-ablation/get-ratings-counterbalanced.err

# Load conda environment
source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
source activate assistant-gate

# navigate to python script parent directory
cd ~/research_projects/assistant-gate/experiments/star-3-mistral-ablation/response-win-rates-counterbalanced/

# python get-ratings.py qa_model=baseline qa_model_2=m0 split=test
python get-ratings.py qa_model=baseline qa_model_2=m1 split=test
python get-ratings.py qa_model=baseline qa_model_2=m2 split=test
python get-ratings.py qa_model=baseline qa_model_2=m3 split=test
# python get-ratings.py qa_model=m0 qa_model_2=baseline split=test
# python get-ratings.py qa_model=m1 qa_model_2=baseline split=test
# python get-ratings.py qa_model=m2 qa_model_2=baseline split=test