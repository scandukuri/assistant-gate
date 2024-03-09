#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops2
#SBATCH --gres=gpu:1  # Requesting four GPUs
#SBATCH --mem=48G 
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=script-logs-2-bsft/get-responses.out
#SBATCH --error=script-logs-2-bsft/get-responses.err

# Load conda environment
source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
source activate assistant-gate

# navigate to python script parent directory
cd ~/research_projects/assistant-gate/experiments/star-2-bsft/response-win-rates/

python get-responses.py answer_model=baseline qa_model=baseline split=test
python get-responses.py answer_model=m0 qa_model=m0 split=test
python get-responses.py answer_model=m1 qa_model=m1 split=test
python get-responses.py answer_model=m2 qa_model=m2 split=test

