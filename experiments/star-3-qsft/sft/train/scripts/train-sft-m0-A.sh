#!/bin/bash

#SBATCH --account=cocoflops                 # Specify the account
#SBATCH --partition=cocoflops               # Specify the partition
#SBATCH --nodelist=cocoflops2          # Request the specific node
#SBATCH --gres=gpu:4                     # Request GPUs
#SBATCH --mem=468GB                         # Memory request
#SBATCH --cpus-per-task=48                 # Number of CPUs per task
#SBATCH --time=256:00:00                    # Time limit
#SBATCH --output=script-logs-3-qsft/m0-A-sft.out
#SBATCH --error=script-logs-3-qsft/m0-A-sft.err

source /scr/andukuri/miniconda3/etc/profile.d/conda.sh
conda activate assistant-gate

cd /sailhome/andukuri/research_projects/assistant-gate/experiments/star-3-qsft/sft/train

export MASTER_ADDR=cocoflops2
python train_sft.py qa_model=m0 model=m0 split=A

