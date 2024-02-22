import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import argparse
import fire
import pandas as pd
from tqdm import tqdm
import numpy as np
import json
import logging
import torch
import random
import re
import time
import os
import signal
from collections import defaultdict
from datasets import load_dataset, Dataset
from utils import *
from transformers import AutoModelForCausalLM, AutoTokenizer

# import models
from AG.models.huggingface.hf_inference_model import HFInferenceModel
from AG.models.openai.azure import AsyncAzureChatLLM
from AG.models.openai.gpt4 import GPT4Agent
from AG.models.vllm_models.inference_model import VLLMInferenceModel


# logging
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="conf", config_name='config')
def main(args: DictConfig) -> None:
    logging.info("Loading raw simulated conversations for preprocessing...")
    
    
    with open(args.iteration.CONVERSATIONS_DIR, 'r') as f:
        conversations = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer.model_config.model)

    
    sources, targets = list(), list()
    for key, lst in conversations.items():
        for conversation in lst:
            conversation = extract_history(conversation)
            turns = create_turns(conversation)
            messages = [{"role": args.ROLES[i % 2], "content": turn} for i, turn in enumerate(turns)]
            for i in range(1, len(messages), 2):
                sources.append(tokenizer.decode(tokenizer.apply_chat_template(messages[:i])))
                targets.append(tokenizer.decode(tokenizer.apply_chat_template(messages[:i + 1])))
    
    
    with open(f"/sailhome/andukuri/research_projects/assistant-gate/experiments/v3/sft/data/sources.json", 'w') as f:
        json.dump(sources, f)
    with open(f"/sailhome/andukuri/research_projects/assistant-gate/experiments/v3/sft/data/targets.json", 'w') as f:
        json.dump(targets, f)
            
    
    return -1


if __name__ == '__main__':
    try:
        fire.Fire(main())
    except:
        pass
