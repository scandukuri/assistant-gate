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
    logging.info(f"Loading models for multi-turn dialogue for {args.split.name}...")
    random.seed(1)
    
    
    # Load prompts
    with open(args.split.PROMPT_DIR, "r") as f:
        prompts = json.load(f)
        prompts = [s.strip() for s in prompts]
    # Load personas
    with open(args.split.PERSONA_DIR, 'r') as f:
        personas = json.load(f)
    # Load names
    with open(args.split.NAMES_DIR, 'r') as f:
        names = json.load(f)
    
    
    if os.path.exists(f"simulated-conversations/{args.qa_model.shortname}_{args.split.name}.json"):
        qa_model = VLLMInferenceModel(**args.qa_model.model_config)
        pulled_conversations = json.load(open(f"simulated-conversations/{args.qa_model.shortname}_{args.split.name}.json", 'r'))
        output_conversations = defaultdict(list)
        for j, persona in enumerate(personas):
            logging.info(f"Beginning simulations for persona {j}...")
            
            # get list of all prompts for that persona in order
            conversations = [pulled_conversations[f'prompt-{p} persona-{j}'] for p in range(len(prompts))]
            flattened_conversations = flatten_list(conversations)
            
            conversation_batches = list(batch_list(flattened_conversations, args.qa_model.run.batch_size))  # Create batches of prompts
            final_conversations = list()
            for batch_index, conversation_batch in enumerate(conversation_batches):
                qa_responses = qa_model.batch_prompt(conversation_batch, **args.qa_model.run.completion_config)
                appended_conversations = [unfinished_conversation + '\n' + qa_response + EOS_TOKEN for unfinished_conversation, qa_response in zip(conversation_batch, qa_responses)]
                final_conversations.extend(appended_conversations)
            final_conversations = chunk_list(final_conversations, args.qa_model.run.initial_completion_config.num_return_sequences)
            for i, sublist in enumerate(final_conversations):
                pair_key = f"prompt-{i} persona-{j}"
                output_conversations[pair_key].extend(sublist)
    
        
        with open(f"simulated-conversations/{args.qa_model.shortname}_{args.split.name}.json", 'w') as f:
            json.dump(output_conversations, f)

    else:
        qa_model = VLLMInferenceModel(**args.qa_model.model_config)
        output_conversations = defaultdict(list)
        for j, persona in enumerate(personas):
            logging.info(f"Beginning simulations for persona {j}...")
            prompt_batches = list(batch_list(prompts, args.qa_model.run.batch_size))  # Create batches of prompts
            for batch_index, prompt_batch in enumerate(prompt_batches):
                logging.info(f"Beginning simulations for persona {j}, prompt batch {batch_index}...")
                # Sample number of turns uniformly from integers [1, MAX_TURNS] inclusive
                
                initial_prompts = [f"{BOS_TOKEN}{B_INST}{QA_PROMPTS[args.QA_PROMPT_IDX].format(names[j], prompt)}{E_INST}" for prompt in prompt_batch]  
                # TODO: Decide the better approach; another simpler way to do this could have been to just duplicate initial_prompts as necessary here, as we do for each flatten_list call in i.e. what is used in the line below, but some people might find existing solution more elegant
                # initial_prompts = flatten_list([[initial_prompt] * args.qa_model.run.initial_completion_config.num_return_sequences for initial_prompt in initial_prompts])
                # Note that changing the solution here will require a change to all the chunking logic later on
                
                
                # QA Model initial turn
                qa_responses = qa_model.batch_prompt(initial_prompts, **args.qa_model.run.initial_completion_config)  # has length num_return_sequences * batch_size
                conversations = list()
                for i, sublist in enumerate(chunk_list(qa_responses, args.qa_model.run.initial_completion_config.num_return_sequences)):
                    conversations.extend([initial_prompts[i] + '\n' + qa_response + EOS_TOKEN for qa_response in sublist])
                conversations = chunk_list(conversations, args.qa_model.run.initial_completion_config.num_return_sequences)
                for i, sublist in enumerate(conversations):
                    prompt_index = batch_index * args.qa_model.run.batch_size + i
                    pair_key = f"prompt-{prompt_index} persona-{j}"
                    output_conversations[pair_key].extend(sublist)
        with open(f"simulated-conversations/{args.qa_model.shortname}_{args.split.name}.json", 'w') as f:
            json.dump(output_conversations, f)


if __name__ == '__main__':
    try:
        fire.Fire(main())
    except:
        pass
