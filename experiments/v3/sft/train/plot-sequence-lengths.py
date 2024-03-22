import os
import logging 
from tqdm import tqdm

import matplotlib.pyplot as plt
import fire
import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer

from utils import *

logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args: DictConfig) -> None:
    logging.info(f"Finding max sequence lengths in preparation for training run...")
    
    # get tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(**args.model.tokenizer_config)
    
    # set padding
    tokenizer.pad_token, tokenizer.padding_side = tokenizer.eos_token, "right"
    tokenizer.special
    
    
    sources, targets = json.load(open(args.data_args.sources_dir, 'r')), json.load(open(args.data_args.targets_dir, 'r'))
    dataset = preprocess(sources=sources, targets=targets, tokenizer=tokenizer)
    lengths = [e['labels'].shape[0] for e in dataset]
    plt.hist(lengths, bins=50, color='#FFA500')
    plt.title('Training Dataset Sequence Lengths')
    plt.xlabel('# Tokens')
    plt.ylabel('Frequency')
    plt.savefig('sequence-lengths.png')
    
    
    
    
    
    
if __name__ == "__main__":
    fire.Fire(main())
    