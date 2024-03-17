#!/bin/bash

#SBATCH --account=cocoflops                 # Specify the account
#SBATCH --partition=cocoflops               # Specify the partition
#SBATCH --nodelist=cocoflops-hgx-1          # Request the specific node
#SBATCH --gres=gpu:4                     # Request GPUs
#SBATCH --mem=468GB                         # Memory request
#SBATCH --cpus-per-task=80               # Number of CPUs per task
#SBATCH --time=256:00:00                    # Time limit
#SBATCH --output=script-logs-3-mistral-ablation/m2-A-sft.out
#SBATCH --error=script-logs-3-mistral-ablation/m2-A-sft.err

source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
conda activate assistant-gate

cd /sailhome/andukuri/research_projects/assistant-gate/experiments/star-3-mistral-ablation/sft/train

# note: model should ALWAYS be m0, we always finetune from base model
# qa_model refers to the current iteration of model which generated our new training split
python train_sft.py qa_model=m2 split=A model=m0 