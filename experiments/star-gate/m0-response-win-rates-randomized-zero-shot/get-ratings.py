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
    logging.info(f"Loading model {args.qa_model.shortname} for response generation and win-rate computation for {args.split.name}...")
    random.seed(1)
    
    rating_llm = AsyncAzureChatLLM(**args.rating_model.model_config.azure_api)
    rating_model = GPT4Agent(llm=rating_llm, **args.rating_model.run.completion_config)
    
    if not os.path.exists(f'{WINRATE_PATH}/{VERSION_2_BSFT}/{args.qa_model.shortname}_{args.qa_model_2.shortname}'):
        os.makedirs(f'{WINRATE_PATH}/{VERSION_2_BSFT}/{args.qa_model.shortname}_{args.qa_model_2.shortname}')
    
    
    # Load personas, prompts
    with open(f"{PERSONAS_PATH}/{VERSION_2_BSFT}/{args.split.name}.json", 'r') as f:
        personas = json.load(f)
    with open(f"{PERSONAS_PATH}/{VERSION_2_BSFT}/{args.split.name}_NAMES.json", 'r') as f:
        names = json.load(f)
    with open(f"{PROMPT_PATH}/{VERSION_2_BSFT}/{args.split.name}.json", "r") as f:
        prompts = json.load(f)
        prompts = [s.strip() for s in prompts]
    
        
    qa_responses_1 = []
    for i in range(1, args.MAX_TURNS + 1):
        qa_responses_1.append(json.load(open(f'{WINRATE_PATH}/{VERSION_2_BSFT}/{args.qa_model.shortname}/{args.split.name}_turn-{i}_m0-responses_zero_shot.json', 'r')))
    
    qa_responses_2 = []
    for i in range(1, args.MAX_TURNS + 1):
        qa_responses_2.append(json.load(open(f'{WINRATE_PATH}/{VERSION_2_BSFT}/{args.qa_model_2.shortname}/{args.split.name}_turn-{i}_m0-responses_zero_shot.json', 'r')))
    
    turns_1, turns_2, turns_3 = random.sample(list(qa_responses_1[0].keys()), args.n//3), random.sample(list(qa_responses_1[1].keys()), args.n//3), random.sample(list(qa_responses_1[2].keys()), args.n//3)
    for t_num, group in enumerate([turns_1, turns_2, turns_3]):
        group_prompt_indices = [int(key[key.find('prompt-') + len('prompt-'):key.find('persona-')].strip()) for key in group]
        group_persona_indices = [int(key[key.find('persona-') + len('persona-'):].strip()) for key in group]
        
        group_prompts = [prompts[idx] for idx in group_prompt_indices]
        group_personas = [personas[idx] for idx in group_persona_indices]
        group_qa_responses_1 = [qa_responses_1[t_num][key] for key in group]
        group_qa_responses_2 = [qa_responses_2[t_num][key] for key in group]
         # Example usage:
        n = len(group_prompts) # For example, to generate a list of 10 coin flips
        flips = generate_coin_flips(n)
        
        group_qa_responses_1, group_qa_responses_2 = zip(*[
            (group_qa_responses_2[i], group_qa_responses_1[i]) if flips[i] == 1 else (group_qa_responses_1[i], group_qa_responses_2[i])
            for i in range(n)
        ])

        rating_prompts = [RATER_MAIN_PROMPTS[args.RATER_MAIN_PROMPT_IDX].format(persona, prompt, qa_response_1, qa_response_2) for persona, prompt, qa_response_1, qa_response_2 in zip(group_personas, group_prompts, group_qa_responses_1, group_qa_responses_2)]
        rating_messages = rating_model.batch_prompt(system_message=RATER_SYS_PROMPTS[args.RATER_SYS_PROMPT_IDX], messages=rating_prompts)
        rating_messages = [msg[0] for msg in rating_messages]
        for msg_idx, msg in enumerate(rating_messages):
            if flips[msg_idx] == 1:
                if msg[msg.find('Final Response:') + len('Final Response:'):].lower().strip() == 'a':
                    rating_messages[msg_idx] = msg[:msg.find('Final Response:')] + 'Final Response: B'
                elif msg[msg.find('Final Response:') + len('Final Response:'):].lower().strip() == 'b':
                    rating_messages[msg_idx] = msg[:msg.find('Final Response:')] + 'Final Response: A'
        
            
        logging.info(f"Turns {t_num + 1}:")
        logging.info(f"Rating messages: ")
        logging.info(rating_messages)
        
        with open(f'{WINRATE_PATH}/{VERSION_2_BSFT}/{args.qa_model.shortname}_{args.qa_model_2.shortname}/{args.split.name}_turn-{t_num + 1}_m0-win-rates-randomized_zero_shot.json', 'w') as f:
            json.dump(dict(zip(group, rating_messages)), f)
        
        
    
    
if __name__ == '__main__':
    try:
        fire.Fire(main())
    except:
        pass
