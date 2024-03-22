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
import copy
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
    logging.info(f"Loading model {args.qa_model.shortname} for response generation and win-rate computation for {args.split.name}...")
    random.seed(1)

    #qa_model = VLLMInferenceModel(**args.qa_model.model_config)
    qa_model = HFInferenceModel(**args.qa_model.model_config)
    
    rating_llm = AsyncAzureChatLLM(**args.rating_model.model_config.azure_api)
    rating_model = GPT4Agent(llm=rating_llm, **args.rating_model.run.completion_config)
    
    if not os.path.exists(f'{WINRATE_PATH}/{VERSION_AG}/{args.qa_model.shortname}'):
        os.makedirs(f'{WINRATE_PATH}/{VERSION_AG}/{args.qa_model.shortname}')
    
    
    # Load personas, prompts
    with open(f"{PERSONAS_PATH}/{VERSION_AG}/{args.split.name}.json", 'r') as f:
        personas = json.load(f)
    with open(f"{PERSONAS_PATH}/{VERSION_AG}/{args.split.name}_NAMES.json", 'r') as f:
        names = json.load(f)
    with open(f"{PROMPT_PATH}/{VERSION_AG}/{args.split.name}.json", "r") as f:
        prompts = json.load(f)
        prompts = [s.strip() for s in prompts]
    with open(f"{GOLD_PATH}/{VERSION_AG}/{args.split.name}.json", "r") as f:
        gold = json.load(f)
        
    turns_conversations = []
    for i in range(1, args.MAX_TURNS + 1):
        turns_conversations.append(json.load(open(f"{SIMULATION_PATH}/{VERSION_AG}/{args.qa_model.shortname}/{args.split.name}_turn-{i}.json", "r")))
    #pooled_conversations = json.load(open(f"{SIMULATION_PATH}/{VERSION_AG}/{args.qa_model.shortname}/{args.split.name}{f'_top-k-{args.k}' if args.k > 0 else ''}.json", "r"))
    pooled_conversations = copy.deepcopy(turns_conversations[0])
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer.model_config.model)
    # turns_1, turns_2, turns_3 = random.sample(list(turns_conversations[0].keys()), args.n//3), random.sample(list(turns_conversations[1].keys()), args.n//3), random.sample(list(turns_conversations[2].keys()), args.n//3)
    turns_1 = list(pooled_conversations.keys())
    # all_qa_responses = list()
    # for t_num, group in enumerate([turns_1, turns_2, turns_3]):
    for t_num, group in enumerate([turns_1]):
        # group is a list of keys f'prompt-{i} persona-{j}' where prompt and persona can be used to index the prompts and personas list above
        conversations = [pooled_conversations[key][0] for key in group]
        
        group_qa_prompts = list()
        group_prompt_indices = [int(key[key.find('prompt-') + len('prompt-'):key.find('persona-')].strip()) for key in group]
        group_persona_indices = [int(key[key.find('persona-') + len('persona-'):].strip()) for key in group]
        for c_idx, conversation in enumerate(conversations):
            conversation = extract_history(conversation)
            
            turns = create_turns(conversation)
            turns[-1] += f'\n\n{turns[0]}'  ## add the prompt to the end of the conversation again to prompt the model to answer, rather than ask another elicitation question
            turns[0] = f"My name is {names[group_persona_indices[c_idx]]}.\n\n{turns[0]}"
            messages = [{"role": args.ROLES[i % 2], "content": turn} for i, turn in enumerate(turns)]
            
            group_qa_prompts.append(tokenizer.decode(tokenizer.apply_chat_template(messages)))
        breakpoint()
        group_qa_responses = qa_model.batch_prompt(group_qa_prompts, **args.qa_model.run.completion_config)
        breakpoint()
        with open(f'{WINRATE_PATH}/{VERSION_AG}/{args.qa_model.shortname}/{args.split.name}_turn-{t_num + 1}_qa_responses.json', 'w') as f:
            json.dump(dict(zip(group, group_qa_responses)), f)
       
    
    
if __name__ == '__main__':
    try:
        fire.Fire(main())
    except:
        pass
