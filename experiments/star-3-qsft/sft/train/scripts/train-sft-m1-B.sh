#!/bin/bash

#SBATCH --account=cocoflops                 # Specify the account
#SBATCH --partition=cocoflops               # Specify the partition
#SBATCH --nodelist=cocoflops2          # Request the specific node
#SBATCH --gres=gpu:4                     # Request GPUs
#SBATCH --mem=400GB                         # Memory request
#SBATCH --cpus-per-task=80                 # Number of CPUs per task
#SBATCH --time=256:00:00                    # Time limit
#SBATCH --output=script-logs-3-qsft/m1-B-sft.out
#SBATCH --error=script-logs-3-qsft/m1-B-sft.err

source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
conda activate assistant-gate

cd /sailhome/andukuri/research_projects/assistant-gate/experiments/star-3-qsft/sft/train

# note: model should ALWAYS be m0, we always finetune from base model
# qa_model refers to the current iteration of model which generated our new training split
python train_sft.py qa_model=m1 split=B model=m0 