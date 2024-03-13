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
    logging.info(f"Loading model for gold QA generation for {args.split.name} split...")
    random.seed(1)
    
    # Load model
    is_openai = "openai" in args.model.model_type.lower()
    is_vllm = "vllm" in args.model.model_type.lower()
    if not (is_openai or is_vllm):
        logging.info("Model type not yet supported.")
        return -1
    
    if is_openai:
        llm = AsyncAzureChatLLM(**args.model.model_config.azure_api)
        model = GPT4Agent(llm=llm, **args.model.run.completion_config)
    elif is_vllm:
        model = VLLMInferenceModel(**args.model.model_config)
        BOS_TOKEN, EOS_TOKEN, B_INST, E_INST = '<s>', '</s>', '[INST]', '[/INST]'
    

    
    # Load personas, prompts
    with open(f"{PERSONAS_PATH}/{VERSION_3_QSFT}/{args.split.name}.json", 'r') as f:
        personas = json.load(f)
    
    with open(f"{PROMPT_PATH}/{VERSION_3_QSFT}/{args.split.name}.json", "r") as f:
        prompts = json.load(f)
        prompts = [s.strip() for s in prompts]

    logging.info(f"{len(personas)} total personas...")
    logging.info(f"{len(prompts)} total prompts...") 
    
    if not os.path.exists(f"{GOLD_PATH}/{VERSION_3_QSFT}"):
        os.makedirs(f"{GOLD_PATH}/{VERSION_3_QSFT}")


    
    
    random.seed(1)
    # Generate test split gold responses
    logging.info(f"Beginning {args.split.name} split...")
    gold_responses = defaultdict(list)       
    for j, persona in enumerate(personas):
        logging.info(f"Generating gold responses for persona {j}...")
        prompt_batches = list(batch_list(prompts, args.model.run.batch_size))  # Create batches of prompts
        
        for batch_index, prompt_batch in enumerate(prompt_batches):
            # Generate a list of formatted prompts for the current batch
            formatted_prompts = [GENERATION_PROMPTS[args.GENERATION_PROMPT_IDX].format(persona, prompt) for prompt in prompt_batch]
            
            if is_openai:
                # Adjust for batch processing
                responses = model.batch_prompt(system_message=SYS_PROMPTS[args.SYS_PROMPT_IDX], messages=formatted_prompts)
                responses = flatten_list(responses)
            elif is_vllm:
                # Adjust for batch processing
                batched_prompt = [f"{BOS_TOKEN}{B_INST} {SYS_PROMPTS[args.SYS_PROMPT_IDX]}\n\n{prompt}{E_INST}" for prompt in formatted_prompts]
                responses = model.batch_prompt(batched_prompt, **args.model.run.completion_config)
            
            
            # Correctly indexing and storing responses for each prompt in the batch
            for i, response in enumerate(responses):
                prompt_index = batch_index * args.model.run.batch_size + i
                gold_responses_key = f"prompt-{prompt_index} persona-{j}"
                if gold_responses_key not in gold_responses:
                    gold_responses[gold_responses_key] = []
                gold_responses[gold_responses_key].extend([response])
        with open(f"{GOLD_PATH}/{VERSION_3_QSFT}/{args.split.name}.json", 'w') as f:
            json.dump(gold_responses, f)
    
    with open(f"{GOLD_PATH}/{VERSION_3_QSFT}/{args.split.name}.json", 'w') as f:
        json.dump(gold_responses, f)
            
    
if __name__ == '__main__':
    try:
        fire.Fire(main())
    except:
        pass