import hydra
from omegaconf import DictConfig
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
    logging.info("Loading GPT-4 and generating personas...")
    random.seed(1)
    
    
    # GPT-4 support only atm
    args.model.model_config.azure_api.api_key = os.getenv("OPENAI_API_KEY")
    llm = AsyncAzureChatLLM(**args.model.model_config.azure_api)
    model = GPT4Agent(llm=llm, **args.model.run.completion_config)

    # train split A
    expert_personas = json.load(open(EXPERT_DIR, 'r'))
    # messages = []
    # for i in range(SHOT_GROUPS):
    #     message = GENERATION_PROMPTS[PROMPT_IDX] + '\n\n'
    #     few_shot_indices = random.sample(range(0, len(expert_personas)), 2)
    #     few_shots = '\n\n'.join([expert_personas[j] for j in few_shot_indices])
    #     message += few_shots + '\n\n'
    #     messages.append(message)
        
    
    # # completions is a list containing len(messages) lists, each of size args.model.run.completion_config.n
    # completions = model.batch_prompt(
    #                 system_message=SYS_MSGS[SYS_IDX],
    #                 messages=messages,
    #             )
    # flattened_completions = [item for sublist in completions for item in sublist]
    
    
    # with open(OUT_DIR.format('public', 'A', SYS_IDX, PROMPT_IDX, args.model.run.completion_config.temperature, args.model.run.completion_config.top_p, args.model.run.completion_config.n, SHOT_GROUPS), 'w') as f:
    #     json.dump(flattened_completions, f)
    
    
    # train split B
    random.seed(1)
    messages = []
    for i in range(SHOT_GROUPS):
        message = GENERATION_PROMPTS[PROMPT_IDX] + '\n\n'
        few_shot_indices = random.sample(range(0, len(expert_personas)), 2)
        few_shots = '\n\n'.join([expert_personas[j] for j in few_shot_indices])
        message += few_shots + '\n\n'
        messages.append(message)
        
    
    # completions is a list containing len(messages) lists, each of size args.model.run.completion_config.n
    completions = model.batch_prompt(
                    system_message=SYS_MSGS[SYS_IDX],
                    messages=messages,
                )
    flattened_completions = [item for sublist in completions for item in sublist]
    
    
    with open(OUT_DIR.format('public', 'B', SYS_IDX, PROMPT_IDX, args.model.run.completion_config.temperature, args.model.run.completion_config.top_p, args.model.run.completion_config.n, SHOT_GROUPS), 'w') as f:
        json.dump(flattened_completions, f)
    
    
    # test split
    messages = []
    for i in range(TEST_SHOT_GROUPS):
        message = GENERATION_PROMPTS[PROMPT_IDX] + '\n\n'
        few_shot_indices = random.sample(range(0, len(expert_personas)), 2)
        few_shots = '\n\n'.join([expert_personas[j] for j in few_shot_indices])
        message += few_shots + '\n\n'
        messages.append(message)
        
    
    # completions is a list containing len(messages) lists, each of size args.model.run.completion_config.n
    completions = model.batch_prompt(
                    system_message=SYS_MSGS[SYS_IDX],
                    messages=messages,
                )
    flattened_completions = [item for sublist in completions for item in sublist]
    
    
    with open(TEST_OUT_DIR.format('private', SYS_IDX, PROMPT_IDX, args.model.run.completion_config.temperature, args.model.run.completion_config.top_p, args.model.run.completion_config.n, TEST_SHOT_GROUPS), 'w') as f:
        json.dump(flattened_completions, f)
    
    
    
    


if __name__ == '__main__':
    try:
        fire.Fire(main())
    except:
        pass