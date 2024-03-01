## GENERATE QA:

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

from utils import *
from paths import *


# import models
from AG.models.huggingface.hf_inference_model import HFInferenceModel
from AG.models.openai.azure import AsyncAzureChatLLM
from AG.models.openai.gpt4 import GPT4Agent
from AG.models.vllm_models.inference_model import VLLMInferenceModel


# logging
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="conf", config_name='config')
def main(args: DictConfig) -> None:
    logging.info(f"Loading models for multi-turn dialogue for {args.split.name}...")
    random.seed(1)
    
    if args.turn.number > args.MAX_TURNS:
        logging.info(f"Cannot exceed {args.MAX_TURNS} turns per conversation. Exiting.")
        return
    
    # Load prompts
    with open(f'{PROMPT_PATH}/{VERSION}/{args.split.name}.json', "r") as f:
        prompts = json.load(f)
        prompts = [s.strip() for s in prompts]
    # Load personas
    with open(f'{PERSONAS_PATH}/{VERSION}/{args.split.name}.json', "r") as f:
        personas = json.load(f)
    # Load names
    with open(f'{PERSONAS_PATH}/{VERSION}/{args.split.name}_NAMES.json', "r") as f:
        names = json.load(f)
    
    if not os.path.exists(f"{SIMULATION_PATH}/{VERSION}/{args.qa_model.shortname}"):
        os.makedirs(f"{SIMULATION_PATH}/{VERSION}/{args.qa_model.shortname}")
    
    
    
    qa_model = VLLMInferenceModel(**args.qa_model.model_config)
    breakpoint()


if __name__ == '__main__':
    try:
        fire.Fire(main())
    except:
        pass


