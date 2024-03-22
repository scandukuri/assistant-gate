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
    logging.info("Loading model for experimental group likelihood generation...")
    random.seed(1)
    
    
    # Load model
    is_vllm = "vllm" in args.model.model_type.lower()
    if not (is_vllm):
        logging.info("Model type not yet supported.")
        return -1
    if is_vllm:
        model = VLLMInferenceModel(**args.model.model_config)
   
    is_mistral = True if 'mixtral' in args.model.name or 'mistral' in args.model.name else False
    if is_mistral:    
        BOS_TOKEN, EOS_TOKEN, B_INST, E_INST = '<s>', '</s>', '[INST]', '[/INST]'
        
    
    # Load prompts
    with open(f"{PROMPT_PATH}/{VERSION}/{args.split.name}.json", "r") as f:
        prompts = json.load(f)
        prompts = [s.strip() for s in prompts]
    
    # Load personas
    with open(f"{PERSONAS_PATH}/{VERSION}/{args.split.name}.json", 'r') as f:
        personas = json.load(f)
    
    # Load names
    with open(f"{PERSONAS_PATH}/{VERSION}/{args.split.name}_NAMES.json", 'r') as f:
        names = json.load(f)
    

    with open(f'{GOLD_PATH}/{VERSION}/{args.split.name}.json', 'r') as f:
        gold_responses = json.load(f)
        
    final_log_probs = defaultdict(list)
    for j, persona in enumerate(personas):
        for i, prompt in enumerate(prompts):
            if i % 100 == 0:
                logging.info(f'CHECKPOINT: Computing batched log probabilities for prompt-{i} persona-{j}...')
            if is_vllm:
                if is_mistral:
                    log_probs = model.batch_log_probs(
                        prompts=[f"{BOS_TOKEN}{B_INST}My name is {names[j]}.\n\n{prompt}{E_INST}\nFINAL ANSWER: "], 
                        answers=[f"{BOS_TOKEN}{B_INST}My name is {names[j]}.\n\n{prompt}{E_INST}\nFINAL ANSWER: {gold_responses[f'prompt-{i} persona-{j}'][0]}"]
                    ).to('cpu')
                    final_log_probs[f'prompt-{i} persona-{j}'].extend(log_probs.tolist())
                    

    if not os.path.exists(f"{LOGPROBS_PATH}/{VERSION}/{args.condition.name}/{args.qa_model.shortname}"):
        os.makedirs(f"{LOGPROBS_PATH}/{VERSION}/{args.condition.name}/{args.qa_model.shortname}")
    with open(f"{LOGPROBS_PATH}/{VERSION}/{args.condition.name}/{args.qa_model.shortname}/{args.split.name}.json", 'w') as f:
        json.dump(final_log_probs, f)
    


if __name__ == '__main__':
    try:
        fire.Fire(main())
    except:
        pass