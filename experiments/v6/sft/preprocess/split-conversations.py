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


# logging
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="conf", config_name='config')
def main(args: DictConfig) -> None:
    logging.info("Loading raw simulated conversations for preprocessing...")
    
    # TODO: fix directory
    

    with open(f"{SIMULATION_PATH}/{VERSION}/{args.qa_model.shortname}/{args.split.name}.json", 'r') as f:
        conversations = json.load(f)
    ### THIS DIRECTORY IS WRONG FIX IT
    with open(f"{GOLD_PATH}/{VERSION}/{args.qa_model.shortname}/{args.split.name}.json", 'r') as f:
        gold_responses = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer.model_config.model)

    
    targets = list()
    for key, lst in conversations.items():
        gold = gold_responses[key]
        for conversation in lst:
            conversation = extract_history(conversation)
            turns = create_turns(conversation)
            turns[-1] += f'\n\n{turns[0]}'  ## add the prompt to the end of the conversation again to prompt the model to answer
            turns.append(gold)
            messages = [{"role": args.ROLES[i % 2], "content": turn} for i, turn in enumerate(turns)]
            targets.append(messages)
    
    if not os.path.exists(f"{SFT_DATA_PATH}/{VERSION}/{args.iteration.shortname}"):
        os.makedirs(f"{SFT_DATA_PATH}/{VERSION}/{args.iteration.shortname}")
    with open(f"{SFT_DATA_PATH}/{VERSION}/{args.iteration.shortname}/{args.split.name}_targets.json", 'w') as f:
        json.dump(targets, f)
            
    
    return -1


if __name__ == '__main__':
    try:
        fire.Fire(main())
    except:
        pass
