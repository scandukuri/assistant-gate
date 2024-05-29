#!/bin/bash

#SBATCH --account=cocoflops                 # Specify the account
#SBATCH --partition=cocoflops               # Specify the partition
#SBATCH --nodelist=cocoflops-hgx-1          # Request the specific node
#SBATCH --gres=gpu:8                     # Request GPUs
#SBATCH --mem=956GB                         # Memory request
#SBATCH --cpus-per-task=96                 # Number of CPUs per task
#SBATCH --time=256:00:00                    # Time limit
#SBATCH --output=script-logs-llama3/m0-A-sft.out
#SBATCH --error=script-logs-llama3/m0-A-sft.err

source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
conda activate assistant-gate

cd /sailhome/andukuri/research_projects/assistant-gate/experiments/star-gate-llama3/sft/train

python train_sft_NEW.py qa_model=m0 model=m0 split=A
