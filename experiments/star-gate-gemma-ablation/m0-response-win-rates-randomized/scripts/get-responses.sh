#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops-hgx-1
#SBATCH --gres=gpu:4  # Requesting four GPUs
#SBATCH --mem=400G 
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH --output=script-logs-2-gemma-ablation/get-m0-responses.out
#SBATCH --error=script-logs-2-gemma-ablation/get-m0-responses.err

# Load conda environment
source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
source activate assistant-gate

# navigate to python script parent directory
cd ~/research_projects/assistant-gate/experiments/star-2-gemma-ablation/m0-response-win-rates-randomized/

python get-responses.py answer_model=baseline qa_model=baseline split=test
#python get-responses.py answer_model=m0 qa_model=m0 split=test
python get-responses.py answer_model=m0 qa_model=m1 split=test
python get-responses.py answer_model=m0 qa_model=m2 split=test
python get-responses.py answer_model=m0 qa_model=m3 split=test
