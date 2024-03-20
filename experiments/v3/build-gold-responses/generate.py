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


# logging
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="conf", config_name='config')
def main(args: DictConfig) -> None:
    logging.info("Loading model for gold QA generation...")
    random.seed(1)
    
    # Load model
    is_openai = "openai" in args.model.model_type.lower()
    is_vllm = "vllm" in args.model.model_type.lower()
    if not (is_openai or is_vllm):
        logging.info("Model type not yet supported.")
        return -1
    
    if is_openai:
        args.model.model_config.azure_api.api_key = os.getenv("OPENAI_API_KEY")
        human_llm = AsyncAzureChatLLM(**args.model.model_config.azure_api)
        human_model = GPT4Agent(llm=human_llm, **args.model.run.completion_config)
    elif is_vllm:
        model = VLLMInferenceModel(**args.model.model_config)
        BOS_TOKEN, EOS_TOKEN, B_INST, E_INST = '<s>', '</s>', '[INST]', '[/INST]'
    
        
    
    # # Load split A personas, prompts
    # with open(PERSONAS_DIR.format('A'), 'r') as f:
    #     A_personas = json.load(f)
    
    # with open(PROMPTS_DIR.format('A'), "r") as f:
    #     A_prompts = json.load(f)
    #     A_prompts = [s.strip() for s in A_prompts]
    
    
    # # Load split B personas, prompts
    # with open(PERSONAS_DIR.format('B'), 'r') as f:
    #     B_personas = json.load(f)
    
    # with open(PROMPTS_DIR.format('B'), "r") as f:
    #     B_prompts = json.load(f)
    #     B_prompts = [s.strip() for s in B_prompts]
    
    
    # Load test personas, prompts
    with open(TEST_PERSONAS_DIR, 'r') as f:
        test_personas = json.load(f)
    
    with open(TEST_PROMPTS_DIR, "r") as f:
        test_prompts = json.load(f)
        test_prompts = [s.strip() for s in test_prompts]
    
    # logging.info(f"{len(A_personas) + len(B_personas) + len(test_personas)} total personas...")
    # logging.info(f"{len(A_prompts) + len(B_prompts) + len(test_prompts)} total prompts...") 
    
    
    
    # # Generate split A gold responses
    # split = 'A'
    # logging.info(f"Beginning split {split}...")
    # gold_responses = defaultdict(list)       
    # for j, persona in enumerate(A_personas):
    #     logging.info(f"Generating gold responses for persona {j}...")
    #     prompt_batches = list(batch_list(A_prompts, args.model.run.batch_size))  # Create batches of prompts
        
    #     for batch_index, prompt_batch in enumerate(prompt_batches):
    #         # Generate a list of formatted prompts for the current batch
    #         formatted_prompts = [GENERATION_PROMPTS[GENERATION_PROMPT_IDX].format(persona, prompt) for prompt in prompt_batch]
            
    #         if is_openai:
    #             # Adjust for batch processing
    #             responses = model.batch_prompt(system_message=SYS_PROMPTS[SYS_PROMPT_IDX], messages=formatted_prompts)
    #             responses = flatten_list(responses)
    #         elif is_vllm:
    #             # Adjust for batch processing
    #             batched_prompt = [f"{BOS_TOKEN}{B_INST} {SYS_PROMPTS[SYS_PROMPT_IDX]}\n\n{prompt}{E_INST}" for prompt in formatted_prompts]
    #             responses = model.batch_prompt(batched_prompt, **args.model.run.completion_config)
            
    #         # Correctly indexing and storing responses for each prompt in the batch
    #         for i, response in enumerate(responses):
    #             prompt_index = batch_index * args.model.run.batch_size + i
    #             gold_responses_key = f"prompt-{prompt_index} persona-{j}"
    #             if gold_responses_key not in gold_responses:
    #                 gold_responses[gold_responses_key] = []
    #             gold_responses[gold_responses_key].extend([response])
    # with open(f"gold-responses/public/split-{split}_gen-{GENERATION_PROMPT_IDX}_sys-{SYS_PROMPT_IDX}_temp-{args.model.run.completion_config.temperature}.json", 'w') as f:
    #     json.dump(gold_responses, f)
    
    
    # random.seed(1)
    # # Generate split B gold responses
    # split = 'B'
    # logging.info(f"Beginning split {split}...")
    # gold_responses = defaultdict(list)       
    # for j, persona in enumerate(B_personas):
    #     logging.info(f"Generating gold responses for persona {j}...")
    #     prompt_batches = list(batch_list(B_prompts, args.model.run.batch_size))  # Create batches of prompts
        
    #     for batch_index, prompt_batch in enumerate(prompt_batches):
    #         # Generate a list of formatted prompts for the current batch
    #         formatted_prompts = [GENERATION_PROMPTS[GENERATION_PROMPT_IDX].format(persona, prompt) for prompt in prompt_batch]
            
    #         if is_openai:
    #             # Adjust for batch processing
    #             responses = model.batch_prompt(system_message=SYS_PROMPTS[SYS_PROMPT_IDX], messages=formatted_prompts)
    #             responses = flatten_list(responses)
    #         elif is_vllm:
    #             # Adjust for batch processing
    #             batched_prompt = [f"{BOS_TOKEN}{B_INST} {SYS_PROMPTS[SYS_PROMPT_IDX]}\n\n{prompt}{E_INST}" for prompt in formatted_prompts]
    #             responses = model.batch_prompt(batched_prompt, **args.model.run.completion_config)
            
    #         # Correctly indexing and storing responses for each prompt in the batch
    #         for i, response in enumerate(responses):
    #             prompt_index = batch_index * args.model.run.batch_size + i
    #             gold_responses_key = f"prompt-{prompt_index} persona-{j}"
    #             if gold_responses_key not in gold_responses:
    #                 gold_responses[gold_responses_key] = []
    #             gold_responses[gold_responses_key].extend([response])
    # with open(f"gold-responses/public/split-{split}_gen-{GENERATION_PROMPT_IDX}_sys-{SYS_PROMPT_IDX}_temp-{args.model.run.completion_config.temperature}.json", 'w') as f:
    #     json.dump(gold_responses, f)
    
    
    random.seed(1)
    # Generate test split gold responses
    logging.info(f"Beginning test split...")
    gold_responses = defaultdict(list)       
    for j, persona in enumerate(test_personas):
        logging.info(f"Generating gold responses for persona {j}...")
        prompt_batches = list(batch_list(test_prompts, args.model.run.batch_size))  # Create batches of prompts
        
        for batch_index, prompt_batch in enumerate(prompt_batches):
            # Generate a list of formatted prompts for the current batch
            formatted_prompts = [GENERATION_PROMPTS[GENERATION_PROMPT_IDX].format(persona, prompt) for prompt in prompt_batch]
            
            if is_openai:
                # Adjust for batch processing
                responses = model.batch_prompt(system_message=SYS_PROMPTS[SYS_PROMPT_IDX], messages=formatted_prompts)
                responses = flatten_list(responses)
            elif is_vllm:
                # Adjust for batch processing
                batched_prompt = [f"{BOS_TOKEN}{B_INST} {SYS_PROMPTS[SYS_PROMPT_IDX]}\n\n{prompt}{E_INST}" for prompt in formatted_prompts]
                responses = model.batch_prompt(batched_prompt, **args.model.run.completion_config)
            
            # Correctly indexing and storing responses for each prompt in the batch
            for i, response in enumerate(responses):
                prompt_index = batch_index * args.model.run.batch_size + i
                gold_responses_key = f"prompt-{prompt_index} persona-{j}"
                if gold_responses_key not in gold_responses:
                    gold_responses[gold_responses_key] = []
                gold_responses[gold_responses_key].extend([response])
    with open(f"gold-responses/private/gen-{GENERATION_PROMPT_IDX}_sys-{SYS_PROMPT_IDX}_temp-{args.model.run.completion_config.temperature}.json", 'w') as f:
        json.dump(gold_responses, f)
            
    
if __name__ == '__main__':
    try:
        fire.Fire(main())
    except:
        pass