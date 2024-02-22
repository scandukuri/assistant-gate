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


# import models
from AG.models.huggingface.hf_inference_model import HFInferenceModel
from AG.models.openai.azure import AsyncAzureChatLLM
from AG.models.openai.gpt4 import GPT4Agent
from AG.models.vllm_models.inference_model import VLLMInferenceModel


# logging
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="conf", config_name='config')
def main(args: DictConfig) -> None:
    logging.info(f"Loading models for shuffling from {args.split.name}...")
    random.seed(1)
    
    with open(f"simulated-conversations/{args.qa_model.shortname}_{args.split.name}.json", 'r') as f:
        conversations = json.load(f)
    
    output_conversations = shuffle_dict_values(conversations)
    
    with open(f"simulated-conversations/{args.qa_model.shortname}-confusion_{args.split.name}.json", 'w') as f:
        json.dump(output_conversations, f)


if __name__ == '__main__':
    try:
        fire.Fire(main())
    except:
        pass
