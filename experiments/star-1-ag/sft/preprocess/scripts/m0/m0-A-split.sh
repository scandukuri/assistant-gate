#!/bin/bash

#SBATCH --account=cocoflops                 # Specify the account
#SBATCH --partition=cocoflops               # Specify the partition
#SBATCH --nodelist=cocoflops-hgx-1          # Request the specific node
#SBATCH --gres=gpu:0                        # Request GPUs
#SBATCH --mem=64GB                         # Memory request
#SBATCH --cpus-per-task=32                  # Number of CPUs per task
#SBATCH --time=256:00:00                    # Time limit
#SBATCH --output=script-logs/m0-A-split.out
#SBATCH --error=script-logs/m0-A-split.err

source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
conda activate assistant-gate

cd /sailhome/andukuri/research_projects/assistant-gate/experiments/star-1-ag/sft/preprocess

python split-conversations.py iteration=i0 split=A qa_model=m0 tokenizer=m0