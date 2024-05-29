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

# import models
from AG.models.huggingface.hf_inference_model import HFInferenceModel
from AG.models.openai.azure import AsyncAzureChatLLM
from AG.models.openai.gpt4 import GPT4Agent
from AG.models.vllm_models.inference_model import VLLMInferenceModel
from paths import *


# logging
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="conf", config_name='config')
def main(args: DictConfig) -> None:
    logging.info(f"Loading personas for {args.split.name} to check for content violations...")
    random.seed(1)
    
    # Load model
    is_openai = "openai" in args.model.model_type.lower()
    if not (is_openai):
        logging.info("Model type not yet supported.")
        return -1
    
    if is_openai:
        llm = AsyncAzureChatLLM(**args.model.model_config.azure_api)
        model = GPT4Agent(llm=llm, **args.model.run.completion_config)
    
    # Load personas, prompts
    with open(f"{PERSONAS_PATH}/{LLAMA_VERSION}/{args.split.name}.json", 'r') as f:
        personas = json.load(f)
    

    logging.info(f"{len(personas)} total personas...")

    random.seed(1)
    # Generate test split gold responses
    logging.info(f"Beginning {args.split.name} split...")  
    violating_personas = list()  
    for j, persona in enumerate(personas):
        messages = [f'{persona}\nDo you like this persona?']
        try: 
            responses = model.batch_prompt(system_message='', messages=messages)
            responses = flatten_list(responses)
        except:
            violating_personas.append(j, persona)
    
    if len(violating_personas) == 0:
        logging.info(f"No content violations found for personas in {args.split.name} split.")
    
    if not os.path.exists(f"{CONTENT_VIOLATIONS_PATH}/{LLAMA_VERSION}"):
        os.makedirs(f"{CONTENT_VIOLATIONS_PATH}/{LLAMA_VERSION}")
    with open(f"{CONTENT_VIOLATIONS_PATH}/{LLAMA_VERSION}/{args.split.name}.json", 'w') as f:
        json.dump(violating_personas, f)
            
    
if __name__ == '__main__':
    try:
        fire.Fire(main())
    except:
        pass