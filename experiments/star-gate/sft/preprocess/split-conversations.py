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
    

    with open(f"{SIMULATION_PATH}/{VERSION_2_BSFT}/{args.qa_model.shortname}/{args.split.name}_top-k-{args.k}.json", 'r') as f:
        conversations = json.load(f)
    ### THIS DIRECTORY IS WRONG FIX IT
    with open(f"{MODELRESPONSE_PATH}/{VERSION_2_BSFT}/{args.qa_model.shortname}/{args.split.name}_top-k-{args.k}.json", 'r') as f:
        model_responses = json.load(f)
    with open(f'{PERSONAS_PATH}/{VERSION_2_BSFT}/{args.split.name}_NAMES.json', "r") as f:
        names = json.load(f)
    
    targets = list()
    for key, lst in conversations.items():
        logging.info(f"Processing {key}...")
        bsft_responses = model_responses[key]
        p_idx = int(key[key.find('persona-') + len('persona-'):].strip())
        for c_idx, conversation in enumerate(lst):
            
            conversation = extract_history(conversation)
            turns = create_turns(conversation)
            # These two lines specifically for SFT target including model response
            turns[-1] += f'\n\n{turns[0].strip()}'  ## add the prompt to the end of the conversation again to prompt the model to answer
            turns[0] = f"My name is {names[p_idx].strip()}.\n\n{turns[0].strip()}"
            turns.append(bsft_responses[c_idx])  ## add the model response to the end of the conversation
            
            messages = [{"role": args.ROLES[i % 2], "content": turn} for i, turn in enumerate(turns)]
            
            targets.append(messages)

    if not os.path.exists(f"{SFT_DATA_PATH}/{VERSION_2_BSFT}/{args.qa_model.shortname}"):
        os.makedirs(f"{SFT_DATA_PATH}/{VERSION_2_BSFT}/{args.qa_model.shortname}")
    with open(f"{SFT_DATA_PATH}/{VERSION_2_BSFT}/{args.qa_model.shortname}/{args.split.name}.json", 'w') as f:
        json.dump(targets, f)
            
    
    return -1


if __name__ == '__main__':
    try:
        fire.Fire(main())
    except:
        pass

