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
from paths import *

logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args: DictConfig) -> None:
    logging.info(f"Finding max sequence lengths in preparation for training run...")
    
    # get tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(**args.model.tokenizer_config)
    
    # set padding
    tokenizer.pad_token, tokenizer.padding_side = tokenizer.eos_token, "right"
    
    
    targets = json.load(open(f"{SFT_DATA_PATH}/{VERSION_2_BSFT}/{args.qa_model.shortname}/{args.split.name}.json", 'r'))
    breakpoint()
    dataset = preprocess(targets=targets, tokenizer=tokenizer)
    lengths = [e['labels'].shape[0] for e in dataset]
    
    plt.figure(figsize=(12, 6))
    ax = plt.axes()
    ax.set_facecolor("whitesmoke")
    
    plt.hist(lengths, bins=50, color='#1fb37eff')
    plt.title(f'Sequence Lengths for {VERSION_2_BSFT} Split {args.qa_model.shortname} Simulated Conversations on {args.split.name} split')
    plt.xlabel('# Tokens')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.margins(0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.grid(True, color='white', linestyle='-', linewidth=0.1)
    plt.savefig(f'seq-length-plots/{VERSION_2_BSFT} {args.qa_model.shortname}-{args.split.name}-sequence-lengths.png')
    
    
    
    
    
    
if __name__ == "__main__":
    fire.Fire(main())
    
