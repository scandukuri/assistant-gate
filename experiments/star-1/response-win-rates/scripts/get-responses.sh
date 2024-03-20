#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops-hgx-1
#SBATCH --gres=gpu:2  # Requesting four GPUs
#SBATCH --mem=96G 
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=script-logs/get-responses.out
#SBATCH --error=script-logs/get-responses.err

# Load conda environment
source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
source activate assistant-gate

# navigate to python script parent directory
cd ~/research_projects/assistant-gate/experiments/star-1/response-win-rates/

python get-responses.py answer_model=m0 qa_model=m0 split=test tokenizer=m0
python get-responses.py answer_model=m0 qa_model=m1 split=test tokenizer=m0
python get-responses.py answer_model=m0 qa_model=m2 split=test tokenizer=m0
