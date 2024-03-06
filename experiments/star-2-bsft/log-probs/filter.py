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
from paths import *

# import models
from AG.models.huggingface.hf_inference_model import HFInferenceModel
from AG.models.openai.azure import AsyncAzureChatLLM
from AG.models.openai.gpt4 import GPT4Agent
from AG.models.vllm_models.inference_model import VLLMInferenceModel


# logging
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="conf", config_name='config')
def main(args: DictConfig) -> None:
    random.seed(1)
    logging.info(f"Preparing logprobs for filtering top {args.k} conversations per prompt-persona pair...")
    
    
    # Load conversations
    with open(f'{SIMULATION_PATH}/{VERSION_2_BSFT}/{args.qa_model.shortname}/{args.split.name}.json', 'r') as f:
        conversations = json.load(f)
    
    # Load corresponding logprobs
    with open(f"{LOGPROBS_PATH}/{VERSION_2_BSFT}/{args.condition.name}/{args.qa_model.shortname}/{args.split.name}.json", 'r') as f:
        logprobs = json.load(f)
    
    
    filtered_conversations, filtered_indices, filtered_logprobs = defaultdict(list), defaultdict(list), defaultdict(list)
    for i, key in enumerate(list(logprobs.keys())):
        if i % 50 == 0:
            logging.info(f'CHECKPOINT: Filtering log probabilities for {key}...')
        # Get indices of top k logprobs for current key
        max_indices = np.argpartition(logprobs[key], -args.k)[-args.k:]
        max_indices_sorted = max_indices[np.argsort([-logprobs[key][i] for i in max_indices])]  # Sort indices by logprob value

        
        # Filter indices and logprobs based on top k indices
        filtered_indices[key] = max_indices_sorted.tolist()
        filtered_logprobs[key] = [logprobs[key][idx] for idx in max_indices_sorted]
        filtered_conversations[key] = [conversations[key][idx] for idx in max_indices_sorted]

    if not os.path.exists(f"{LOGPROBS_PATH}/{VERSION_2_BSFT}/{args.condition.name}/{args.qa_model.shortname}"):
        os.makedirs(f"{LOGPROBS_PATH}/{VERSION_2_BSFT}/{args.condition.name}/{args.qa_model.shortname}")

    with open(f"{LOGPROBS_PATH}/{VERSION_2_BSFT}/{args.condition.name}/{args.qa_model.shortname}/{args.split.name}_top-k-{args.k}.json", 'w') as f:
        json.dump(filtered_logprobs, f)
    with open(f"{LOGPROBS_PATH}/{VERSION_2_BSFT}/{args.condition.name}/{args.qa_model.shortname}/{args.split.name}_top-k-{args.k}_indices.json", 'w') as f:
        json.dump(filtered_indices, f)
    # THIS LAST ONE IS WRONG, fix later
    if args.condition.name == 'qa-experimental':
        if not os.path.exists(f"{SIMULATION_PATH}/{VERSION_2_BSFT}/{args.qa_model.shortname}"):
            os.makedirs(f"{SIMULATION_PATH}/{VERSION_2_BSFT}/{args.qa_model.shortname}")
        with open(f'{SIMULATION_PATH}/{VERSION_2_BSFT}/{args.qa_model.shortname}/{args.split.name}_top-k-{args.k}.json', 'w') as f:
            json.dump(filtered_conversations, f)
            
    


if __name__ == '__main__':
    try:
        fire.Fire(main())
    except:
        pass