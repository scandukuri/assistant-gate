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
    logging.info(f"Preparing logprobs for filtering top {k} conversations per prompt-persona pair...")
    
    
    # Load indices
    with open(INDICES_DIR, 'r') as f:
        indices = json.load(f)

    # Load corresponding logprobs
    with open(LOGPROBS_DIR, 'r') as f:
        logprobs = json.load(f)
    
    
    filtered_conversations, filtered_logprobs = defaultdict(list), defaultdict(list)
    for i, key in enumerate(list(logprobs.keys())):
        if i % 50 == 0:
            logging.info(f'CHECKPOINT: Filtering log probabilities for {key}...')

        # Filter conversations and logprobs based on top k indices
        filtered_logprobs[key] = [logprobs[key][i] for i in indices[key]]


    with open(LOGPROBS_DIR[:-5] + f'_top-k-{k}.json', 'w') as f:
        json.dump(filtered_logprobs, f)
    


if __name__ == '__main__':
    try:
        fire.Fire(main())
    except:
        pass