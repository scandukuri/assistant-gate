#!/bin/bash

#SBATCH --account=cocoflops                 # Specify the account
#SBATCH --partition=cocoflops               # Specify the partition
#SBATCH --nodelist=cocoflops2          # Request the specific node
#SBATCH --gres=gpu:0                        # Request GPUs
#SBATCH --mem=16GB                         # Memory request
#SBATCH --cpus-per-task=8                 # Number of CPUs per task
#SBATCH --time=256:00:00                    # Time limit
#SBATCH --output=script-logs-1-mistral-ablation/m0-A-sequence-lengths.out
#SBATCH --error=script-logs-1-mistral-ablation/m0-A-sequence-lengths.err

source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
conda activate assistant-gate

cd /sailhome/andukuri/research_projects/assistant-gate/experiments/star-1-mistral-ablation/sft/train/

python plot-sequence-lengths.py split=A model=m0