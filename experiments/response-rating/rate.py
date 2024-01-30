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

from utils import SYSTEM_PROMPTS, SYSTEM_PROMPT_IDX, RATING_PROMPTS, RATING_PROMPT_IDX, CONVERSATIONS_DIR

# import models
from AG.models.huggingface.hf_inference_model import HFInferenceModel
from AG.models.openai.azure import AsyncAzureChatLLM
from AG.models.openai.gpt4 import GPT4Agent
from AG.models.vllm_models.inference_model import VLLMInferenceModel


# logging
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="conf", config_name='config')
def main(args: DictConfig) -> None:
    logging.info("Loading model for conversation rating...")
    random.seed(1)
    
    
    # Load human_model
    is_openai = "openai" in args.rating_model.model_type.lower()
    is_vllm = "vllm" in args.rating_model.model_type.lower()
    if not (is_openai or is_vllm):
        logging.info("Model type not yet supported.")
        return -1
    if is_openai:
        args.rating_model.model_config.azure_api.api_key = os.getenv("OPENAI_API_KEY")
        rating_llm = AsyncAzureChatLLM(**args.rating_model.model_config.azure_api)
        rating_model = GPT4Agent(llm=rating_llm, **args.rating_model.run.completion_config)
    elif is_vllm:
        rating_model = VLLMInferenceModel(**args.rating_model.model_config)
        BOS_TOKEN, EOS_TOKEN, B_INST, E_INST = '<s>', '</s>', '[INST]', '[/INST]'
        
    
    # Load conversations
    with open(CONVERSATIONS_DIR, "r") as f:
        conversations = json.load(f)
        

    final_ratings = defaultdict(list)
    for key in list(conversations.keys()):
        ratings = []
        if is_openai:
            responses = rating_model.batch_prompt(system_message=SYSTEM_PROMPTS[SYSTEM_PROMPT_IDX], messages=[f"{RATING_PROMPTS[RATING_PROMPT_IDX]}\n\n{conversation}" for conversation in conversations[key]],)
            ratings = [int(response[0]) for response in responses]
        elif is_vllm:
            responses = rating_model.batch_prompt([f"{BOS_TOKEN}{B_INST} {SYSTEM_PROMPTS[SYSTEM_PROMPT_IDX]}\n\n{RATING_PROMPTS[RATING_PROMPT_IDX]}\n\n{conversation} {E_INST}" for conversation in conversations[key]], **args.rating_model.run.completion_config)
            ratings = [int(response[0]) for response in responses]

        final_ratings[key] = ratings
    
    with open(f"conversation-ratings/system-{SYSTEM_PROMPT_IDX}_rating-{RATING_PROMPT_IDX}_conversations-{CONVERSATIONS_DIR[CONVERSATIONS_DIR.rfind('/') + 1:-5]}.json", 'w') as f:
        json.dump(final_ratings, f)


if __name__ == '__main__':
    try:
        fire.Fire(main())
    except:
        pass