#!/bin/bash

#SBATCH --account=cocoflops                 # Specify the account
#SBATCH --partition=cocoflops               # Specify the partition
#SBATCH --nodelist=cocoflops-hgx-1          # Request the specific node
#SBATCH --gres=gpu:4                     # Request GPUs
#SBATCH --mem=512GB                         # Memory request
#SBATCH --cpus-per-task=48                  # Number of CPUs per task
#SBATCH --time=256:00:00                    # Time limit
#SBATCH --output=script-logs/m0-A-sft.out
#SBATCH --error=script-logs/m0-A-sft.err

source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
conda activate assistant-gate

cd /sailhome/andukuri/research_projects/assistant-gate/experiments/star-1/sft/train

python train_sft.py model=m0 split=A