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
import gc
import signal
from collections import defaultdict
from datasets import load_dataset, Dataset
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

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
    logging.info(f"Loading models for shuffling from {args.split.name}...")
    random.seed(1)
    
    with open(f"{SIMULATION_PATH}/{VERSION_2_GEMMA_ABLATION}/{args.qa_model.shortname}/{args.split.name}.json", 'r') as f:
        conversations = json.load(f)
    with open(f"{PROMPT_PATH}/{VERSION_2_GEMMA_ABLATION}/{args.split.name}.json", 'r') as f:
        prompts = json.load(f)
    with open(f"{PERSONAS_PATH}/{VERSION_2_GEMMA_ABLATION}/{args.split.name}.json", 'r') as f:
        personas = json.load(f)
    
    output_conversations = dict()
    for i in range(len(prompts)):
        prompt_keys = [f'prompt-{i} persona-{j}' for j in range(len(personas))]
        subset_dict = {k: conversations[k] for k in prompt_keys}
        shuffled_subset_dict = shuffle_dict_values(subset_dict)
        output_conversations |= shuffled_subset_dict
        
    
    with open(f"{SIMULATION_PATH}/{VERSION_2_GEMMA_ABLATION}/{args.qa_model.shortname}/{args.split.name}-shuffled.json", 'w') as f:
        json.dump(output_conversations, f)


if __name__ == '__main__':
    try:
        fire.Fire(main())
    except:
        pass