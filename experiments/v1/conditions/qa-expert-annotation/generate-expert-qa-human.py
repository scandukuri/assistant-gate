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

from utils import QA_PROMPTS, QA_PROMPT_IDX, HUMAN_PROMPTS, HUMAN_PROMPT_IDX, HUMAN_SYS_MSGS, HUMAN_SYS_PROMPT_IDX, NAMES_DIR, PERSONAS_DIR, PROMPTS_DIR, MAX_TURNS, filter_completed_conversations, flatten_list

# import models
from AG.models.huggingface.hf_inference_model import HFInferenceModel
from AG.models.openai.azure import AsyncAzureChatLLM
from AG.models.openai.gpt4 import GPT4Agent
from AG.models.vllm_models.inference_model import VLLMInferenceModel


# logging
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="conf", config_name='config')
def main(args: DictConfig) -> None:
    logging.info("Loading models for multi-turn dialogue...")
    random.seed(1)
    
    
    
    # Load prompts
    with open(PROMPTS_DIR, "r") as f:
        prompts = json.load(f)
        prompts = [s.strip() for s in prompts]
        prompts = [prompts[6], prompts[8]]
    BOS_TOKEN, EOS_TOKEN, B_INST, E_INST = '<s>', '</s>', '[INST]', '[/INST]'
    
    # Load personas
    with open(PERSONAS_DIR, 'r') as f:
        personas = json.load(f)
        personas = personas[1:3] + personas[6:9]
    # Load names
    with open(NAMES_DIR, 'r') as f:
        names = json.load(f)
        names = names[1:3] + names[6:9]
        

    final_conversations = defaultdict(list)
    for i, prompt in enumerate(prompts):
        logging.info(f"Beginning simulations for prompt {i}...")
        if i % 2 == 0:
            with open(f"simulated-conversations/qa-{QA_PROMPT_IDX}_humansys-{HUMAN_SYS_PROMPT_IDX}_human-{HUMAN_PROMPT_IDX}_maxturns-{MAX_TURNS}.json", 'w') as f:
                json.dump(final_conversations, f)
        for j, persona in enumerate(personas):
            initial_prompt = f"{BOS_TOKEN}{B_INST}{QA_PROMPTS[QA_PROMPT_IDX].format(names[j], prompt)}{E_INST}"
            print(initial_prompt)
            qa_responses = [input('Enter your first open-ended question to the user: ')]
            
            conversations = [initial_prompt + '\n' + qa_response + EOS_TOKEN for qa_response in qa_responses]
            turns = 1
            
            while turns < MAX_TURNS:
                human_responses = [f"{BOS_TOKEN}{B_INST} {HUMAN_SYS_MSGS[HUMAN_SYS_PROMPT_IDX]}\n\n{HUMAN_PROMPTS[HUMAN_PROMPT_IDX].format(persona, prompt, qa_response[2:]) }{E_INST}" for qa_response in qa_responses]
                print(human_responses[0])
                
                conversations = [unfinished_conversation + '\n' + B_INST + f" A: {input('Enter your roleplayed response: ')} " + E_INST for unfinished_conversation, human_response in zip(conversations, human_responses)]
                print(conversations[0])
                qa_responses = [input('Enter your next open-ended question to the user, given the above conversation history: ')]
                conversations = [unfinished_conversation + '\n' + qa_response + EOS_TOKEN for unfinished_conversation, qa_response in zip(conversations, qa_responses)]
                turns += 1

            final_conversations[f"prompt-{i} persona-{j}"] = conversations
    
    with open(f"simulated-conversations/qa-{QA_PROMPT_IDX}_humansys-{HUMAN_SYS_PROMPT_IDX}_human-{HUMAN_PROMPT_IDX}_maxturns-{MAX_TURNS}.json", 'w') as f:
        json.dump(final_conversations, f)


if __name__ == '__main__':
    try:
        fire.Fire(main())
    except:
        pass