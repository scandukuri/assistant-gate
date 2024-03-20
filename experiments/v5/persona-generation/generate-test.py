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


@hydra.main(version_base=None, config_path="conf", config_name='test-split-config')
def main(args: DictConfig) -> None:
    logging.info(f"Loading GPT-4 and generating personas for {args.split.name} split...")
    random.seed(1)
    
    
    # GPT-4 support only atm
    args.model.model_config.azure_api.api_key = os.getenv("OPENAI_API_KEY")
    llm = AsyncAzureChatLLM(**args.model.model_config.azure_api)
    model = GPT4Agent(llm=llm, **args.model.run.completion_config)


    expert_personas = json.load(open(args.EXPERT_DIR, 'r'))
    
    
    # test split
    messages = []
    for i in range(args.SHOT_GROUPS):
        message = GENERATION_PROMPTS[args.PROMPT_IDX] + '\n\n'
        few_shot_indices = random.sample(range(0, len(expert_personas)), 2)
        few_shots = '\n\n'.join([expert_personas[j] for j in few_shot_indices])
        message += few_shots + '\n\n'
        messages.append(message)
        
    
    # completions is a list containing len(messages) lists, each of size args.model.run.completion_config.n
    completions = model.batch_prompt(
                    system_message=SYS_MSGS[args.SYS_IDX],
                    messages=messages,
                )
    flattened_completions = [item for sublist in completions for item in sublist]
    
    
    with open(f'new-personas/{args.split.name}.json', 'w') as f:
        json.dump(flattened_completions, f)
    
    
    
    


if __name__ == '__main__':
    try:
        fire.Fire(main())
    except:
        pass