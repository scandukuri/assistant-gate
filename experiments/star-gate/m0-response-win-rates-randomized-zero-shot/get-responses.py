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
from transformers import AutoTokenizer

from paths import *
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
    logging.info(f"Loading model {args.qa_model.shortname} conversations for response generation and win-rate computation for {args.split.name}...")
    random.seed(1)

    answer_model = VLLMInferenceModel(**args.answer_model.model_config)
    
    
    if not os.path.exists(f'{WINRATE_PATH}/{VERSION_2_BSFT}/{args.qa_model.shortname}'):
        os.makedirs(f'{WINRATE_PATH}/{VERSION_2_BSFT}/{args.qa_model.shortname}')
    
    
    # Load personas, prompts
    with open(f"{PERSONAS_PATH}/{VERSION_2_BSFT}/{args.split.name}.json", 'r') as f:
        personas = json.load(f)
    with open(f"{PERSONAS_PATH}/{VERSION_2_BSFT}/{args.split.name}_NAMES.json", 'r') as f:
        names = json.load(f)
    with open(f"{PROMPT_PATH}/{VERSION_2_BSFT}/{args.split.name}.json", "r") as f:
        prompts = json.load(f)
        prompts = [s.strip() for s in prompts]
    with open(f"{GOLD_PATH}/{VERSION_2_BSFT}/{args.split.name}.json", "r") as f:
        gold = json.load(f)
        
    turns_conversations = []
    for i in range(1, args.MAX_TURNS + 1):
        turns_conversations.append(json.load(open(f"{SIMULATION_PATH}/{VERSION_2_BSFT}/{args.qa_model.shortname}/{args.split.name}_turn-{i}.json", "r")))
    pooled_conversations = json.load(open(f"{SIMULATION_PATH}/{VERSION_2_BSFT}/{args.qa_model.shortname}/{args.split.name}{f'_top-k-{args.k}' if args.k > 0 else ''}.json", "r"))
    
    
    tokenizer = AutoTokenizer.from_pretrained(**args.qa_model.tokenizer_config)
    turns_1, turns_2, turns_3 = random.sample(list(turns_conversations[0].keys()), args.n//3), random.sample(list(turns_conversations[1].keys()), args.n//3), random.sample(list(turns_conversations[2].keys()), args.n//3)
    all_qa_responses = list()
    for t_num, group in enumerate([turns_1, turns_2, turns_3]):
        # group is a list of keys f'prompt-{i} persona-{j}' where prompt and persona can be used to index the prompts and personas list above
        conversations = [random.choice(pooled_conversations[key]) for key in group]
        print(len(pooled_conversations[group[0]]), len(conversations))
        
        group_answer_prompts = list()
        group_prompt_indices = [int(key[key.find('prompt-') + len('prompt-'):key.find('persona-')].strip()) for key in group]
        group_persona_indices = [int(key[key.find('persona-') + len('persona-'):].strip()) for key in group]
        for c_idx, conversation in enumerate(conversations):
            conversation = extract_history(conversation)
            
            turns = create_turns(conversation)
            turns[-1] += f'\n\n{turns[0]}'  ## add the prompt to the end of the conversation again to prompt the model to answer, rather than ask another elicitation question
            turns[0] = f"My name is {names[group_persona_indices[c_idx]]}.\n\n{turns[0]}"
            messages = [{"role": args.ROLES[i % 2], "content": turn} for i, turn in enumerate(turns)]
            
            group_answer_prompts.append(tokenizer.decode(tokenizer.apply_chat_template(messages)))
        
        group_answer_responses = answer_model.batch_prompt(group_answer_prompts, **args.answer_model.run.completion_config)
        
        with open(f'{WINRATE_PATH}/{VERSION_2_BSFT}/{args.qa_model.shortname}/{args.split.name}_turn-{t_num + 1}_m0-responses_zero_shot.json', 'w') as f:
            json.dump(dict(zip(group, group_answer_responses)), f)
       
    
    
if __name__ == '__main__':
    try:
        fire.Fire(main())
    except:
        pass
