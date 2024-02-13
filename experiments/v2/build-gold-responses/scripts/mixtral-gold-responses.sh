#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops-hgx-1
#SBATCH --gres=gpu:4  # Requesting four GPUs
#SBATCH --mem=468G 
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH --output=job_output.%j.out
#SBATCH --error=job_output.%j.err

# Load conda environment
source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
source activate assistant-gate

# install necessary env packages
cd ~/research_projects/assistant-gate
pip install -e .

# navigate to python script parent directory
cd /sailhome/andukuri/research_projects/assistant-gate/experiments/v2/build-gold-responses

python generate.py model=mixtral-8x7b-instruct-vllm