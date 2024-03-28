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
    logging.info(f"Loading model for {args.condition.name} group likelihood generation...")
    random.seed(1)
    
    
    # # Load model
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
    with open(f"{PROMPT_PATH}/{VERSION_1_ESFT}/{args.split.name}.json", "r") as f:
        prompts = json.load(f)
        prompts = [s.strip() for s in prompts]
    
    # Load personas
    with open(f"{PERSONAS_PATH}/{VERSION_1_ESFT}/{args.split.name}.json", 'r') as f:
        personas = json.load(f)
    
    # Load names
    with open(f"{PERSONAS_PATH}/{VERSION_1_ESFT}/{args.split.name}_NAMES.json", 'r') as f:
        names = json.load(f)
        
    # Load conversations
    with open(f'{SIMULATION_PATH}/{VERSION_1_ESFT}/{args.qa_model.shortname}/{args.split.name}-shuffled.json', 'r') as f:
        all_conversations = json.load(f)
    
    # Load gold answers
    with open(f'{GOLD_PATH}/{VERSION_1_ESFT}/{args.split.name}.json', 'r') as f:
        gold_responses = json.load(f)
        
    
    final_log_probs = defaultdict(list)
    for j, persona in enumerate(personas):
        for i, prompt in enumerate(prompts):
            if i % 100 == 0:
                logging.info(f'CHECKPOINT: Computing batched log probabilities for prompt-{i} persona-{j}...')
            conversations = all_conversations[f"prompt-{i} persona-{j}"]
            
            # split conversations in half not to overload GPU memory
            first, second, third, fourth = conversations[:len(conversations)//4], conversations[len(conversations)//4:2 * len(conversations)//4], conversations[2 * len(conversations)//4:3 * len(conversations)//4], conversations[3 * len(conversations)//4:]

            if is_vllm:
                if is_mistral:
                    for lst in [first, second, third, fourth]:
                        if len(lst) == 0: continue
                        log_probs = model.batch_log_probs(
                            prompts=[f"{BOS_TOKEN}{B_INST}My name is {names[j].strip()}.\n\n{prompt.strip()}{E_INST}{conversation[conversation.find(E_INST) + 7 : conversation.rfind(E_INST)].strip()}\n\n{prompt.strip()}{E_INST}" for conversation in lst], 
                            answers=[f"{BOS_TOKEN}{B_INST}My name is {names[j].strip()}.\n\n{prompt.strip()}{E_INST}{conversation[conversation.find(E_INST) + 7 : conversation.rfind(E_INST)].strip()}\n\n{prompt.strip()}{E_INST}{gold_responses[f'prompt-{i} persona-{j}'][0].strip()}" for conversation in lst]
                            ).to('cpu')
                        final_log_probs[f'prompt-{i} persona-{j}'].extend(log_probs.tolist())


    if not os.path.exists(f"{LOGPROBS_PATH}/{VERSION_1_ESFT}/{args.condition.name}/{args.qa_model.shortname}"):
        os.makedirs(f"{LOGPROBS_PATH}/{VERSION_1_ESFT}/{args.condition.name}/{args.qa_model.shortname}")
    with open(f"{LOGPROBS_PATH}/{VERSION_1_ESFT}/{args.condition.name}/{args.qa_model.shortname}/{args.split.name}-shuffled.json", 'w') as f:
        json.dump(final_log_probs, f)
        

if __name__ == '__main__':
    try:
        fire.Fire(main())
    except:
        pass
