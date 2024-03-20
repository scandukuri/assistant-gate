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

from utils import CONVERSATIONS_DIR, NAMES_DIR, PERSONAS_DIR, PROMPTS_DIR, GOLD_DIR, flatten_list

# import models
from AG.models.huggingface.hf_inference_model import HFInferenceModel
from AG.models.openai.azure import AsyncAzureChatLLM
from AG.models.openai.gpt4 import GPT4Agent
from AG.models.vllm_models.inference_model import VLLMInferenceModel


# logging
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="conf", config_name='config')
def main(args: DictConfig) -> None:
    logging.info("Loading model for positive control likelihood generation...")
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
    with open(PROMPTS_DIR, "r") as f:
        prompts = json.load(f)
        prompts = [s.strip() for s in prompts]

    
    # Load personas
    with open(PERSONAS_DIR, 'r') as f:
        personas = json.load(f)
    
    # Load names
    with open(NAMES_DIR, 'r') as f:
        names = json.load(f)
        
     # Load conversations
    with open(CONVERSATIONS_DIR, 'r') as f:
        all_conversations = json.load(f)
    
    # Load gold answers
    with open(GOLD_DIR, 'r') as f:
        gold_responses = json.load(f)
        
    
    final_log_probs = defaultdict(list)
    for j, persona in enumerate(personas):
        for i, prompt in enumerate(prompts):
            if i % 100 == 0:
                logging.info(f'CHECKPOINT: Computing batched log probabilities for prompt-{i} persona-{j}...')
            conversations = all_conversations[f"prompt-{i} persona-{j}"]
            first, second = conversations[:len(conversations)//2], conversations[len(conversations)//2:]
            if is_vllm:
                if is_mistral:
                    for k,lst in enumerate([first, second]):
                    
                        log_probs = model.batch_log_probs(
                            prompts=[f"{BOS_TOKEN}{B_INST}My name is {names[j]}. Here is some information about me:\n\n{persona}\n\n{prompt}{E_INST}{conversation[conversation.find(E_INST) + 7 : conversation.rfind(E_INST) + 7]}\nFINAL ANSWER: " for conversation in lst], 
                            answers=[f"{BOS_TOKEN}{B_INST}My name is {names[j]}. Here is some information about me:\n\n{persona}\n\n{prompt}{E_INST}{conversation[conversation.find(E_INST) + 7 : conversation.rfind(E_INST) + 7]}\nFINAL ANSWER: {gold_responses[f'prompt-{i} persona-{j}'][0]}" for conversation in lst]
                            ).to('cpu')
                        final_log_probs[f'prompt-{i} persona-{j}'].extend(log_probs.tolist())
    
    with open(f"log-probs/logprobs-poscontrol-{CONVERSATIONS_DIR[CONVERSATIONS_DIR.rfind('/') + 1:]}", 'w') as f:
        json.dump(final_log_probs, f)
    
#log_probs = model.batch_log_probs(prompts=[f"{BOS_TOKEN}{B_INST}My name is {names[j]}. Here is some information about me:\n\n{persona}\n\n{prompt}{E_INST}{conversation[conversation.find(E_INST) + 7 : conversation.rfind(E_INST) + 7]}\nFINAL ANSWER: " for conversation in lst], answers=[f"{BOS_TOKEN}{B_INST}My name is {names[j]}. Here is some information about me:\n\n{persona}\n\n{prompt}{E_INST}{conversation[conversation.find(E_INST) + 7 : conversation.rfind(E_INST) + 7]}\nFINAL ANSWER: {gold_responses[f'prompt-{i} persona-{j}'][0]}" for conversation in lst]).to('cpu')

if __name__ == '__main__':
    try:
        fire.Fire(main())
    except:
        pass