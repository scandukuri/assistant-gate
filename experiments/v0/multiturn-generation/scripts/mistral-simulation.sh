#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH -w cocoflops-hgx-1
#SBATCH --gres=gpu:1  # Requesting one GPUs
#SBATCH --mem=128G 
#SBATCH --cpus-per-task=16
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
cd ~/research_projects/assistant-gate/experiments/multiturn-generation

torchrun --nproc_per_node 1 --master_port 0 generate.py qa_model=mistral-7b-instruct-v02-hf human_model=mixtral-8x7b-instruct-vllm