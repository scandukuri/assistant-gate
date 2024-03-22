#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops-hgx-1
#SBATCH --gres=gpu:0  # Requesting four GPUs
#SBATCH --mem=32G 
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=script-logs-1-mistral-ablation/get-m0-ratings-randomized.out
#SBATCH --error=script-logs-1-mistral-ablation/get-m0-ratings-randomized.err

# Load conda environment
source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
source activate assistant-gate

# navigate to python script parent directory
cd ~/research_projects/assistant-gate/experiments/star-1-mistral-ablation/m0-response-win-rates-randomized/

# python get-ratings.py qa_model=baseline qa_model_2=m0 split=test
python get-ratings.py qa_model=baseline qa_model_2=m1 split=test
python get-ratings.py qa_model=baseline qa_model_2=m2 split=test
python get-ratings.py qa_model=baseline qa_model_2=m3 split=test
# python get-ratings.py qa_model=m0 qa_model_2=baseline split=test
# python get-ratings.py qa_model=m1 qa_model_2=baseline split=test
# python get-ratings.py qa_model=m2 qa_model_2=baseline split=test
