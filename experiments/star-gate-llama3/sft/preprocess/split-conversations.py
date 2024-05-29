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
import copy
from collections import defaultdict
from datasets import load_dataset, Dataset
from utils import *
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    logging.info("Loading raw simulated conversations for preprocessing...")
    
    # TODO: fix directory
    

    with open(f"{MODELRESPONSE_PATH}/{LLAMA_VERSION}/{args.qa_model.shortname}/{args.split.name}_top-k-{args.k}.json", 'r') as f:
        conversations = json.load(f)
    with open(f'{PERSONAS_PATH}/{LLAMA_VERSION}/{args.split.name}.json', "r") as f:
        personas = json.load(f)
    with open(f'{PERSONAS_PATH}/{LLAMA_VERSION}/{args.split.name}_NAMES.json', "r") as f:
        names = json.load(f)
    # Load prompts
    with open(f'{PROMPT_PATH}/{LLAMA_VERSION}/{args.split.name}.json', "r") as f:
        prompts = json.load(f)
        prompts = [s.strip() for s in prompts]
    targets = list()
    for key, lst in conversations.items():
        logging.info(f"Processing {key}...")
        prompt_idx = int(key[key.find('prompt-') + len('prompt-'):key.find('persona-')].strip())
        persona_idx = int(key[key.find('persona-') + len('persona-'):].strip())
        for c_idx, conversation in enumerate(lst):
            final_conv = copy.deepcopy(conversation)
            final_conv[0]['content'] = f"My name is {names[persona_idx].strip()}.\n\n{prompts[prompt_idx].strip()}"
            targets.append(final_conv)

    if not os.path.exists(f"{SFT_DATA_PATH}/{LLAMA_VERSION}/{args.qa_model.shortname}"):
        os.makedirs(f"{SFT_DATA_PATH}/{LLAMA_VERSION}/{args.qa_model.shortname}")
    with open(f"{SFT_DATA_PATH}/{LLAMA_VERSION}/{args.qa_model.shortname}/{args.split.name}.json", 'w') as f:
        json.dump(targets, f) 
    
    return -1


if __name__ == '__main__':
    try:
        fire.Fire(main())
    except:
        pass

