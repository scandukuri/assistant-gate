#!/bin/bash

#SBATCH --account=cocoflops                 # Specify the account
#SBATCH --partition=cocoflops               # Specify the partition
#SBATCH --nodelist=cocoflops2        # Request the specific node
#SBATCH --gres=gpu:0                        # Request GPUs
#SBATCH --mem=64GB                         # Memory request
#SBATCH --cpus-per-task=32                  # Number of CPUs per task
#SBATCH --time=256:00:00                    # Time limit
#SBATCH --output=script-logs-2-mistral-ablation/m2-A-split.out
#SBATCH --error=script-logs-2-mistral-ablation/m2-A-split.err

source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
conda activate assistant-gate

cd /sailhome/andukuri/research_projects/assistant-gate/experiments/star-2-mistral-ablation/sft/preprocess

python split-conversations.py split=A qa_model=m2