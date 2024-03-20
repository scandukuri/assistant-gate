#!/bin/bash

#SBATCH --account=cocoflops                 # Specify the account
#SBATCH --partition=cocoflops               # Specify the partition
#SBATCH --nodelist=cocoflops2          # Request the specific node
#SBATCH --gres=gpu:2                        # Request GPUs
#SBATCH --mem=256GB                         # Memory request
#SBATCH --cpus-per-task=32                  # Number of CPUs per task
#SBATCH --time=256:00:00                    # Time limit
#SBATCH --output=sft.out
#SBATCH --error=sft.err

source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
conda activate assistant-gate

cd /sailhome/andukuri/research_projects/assistant-gate/experiments/v8/sft/train

python train_sft.py model=m0 split=A