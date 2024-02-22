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
# QA_PROMPTS, QA_PROMPT_IDX, HUMAN_PROMPTS, HUMAN_PROMPT_IDX, HUMAN_SYS_MSGS, HUMAN_SYS_PROMPT_IDX, NAMES_DIR, PERSONAS_DIR, PROMPTS_DIR, MAX_TURNS, flatten_list, batch_list, chunk_list, extract_history

# import models
from AG.models.huggingface.hf_inference_model import HFInferenceModel
from AG.models.openai.azure import AsyncAzureChatLLM
from AG.models.openai.gpt4 import GPT4Agent
from AG.models.vllm_models.inference_model import VLLMInferenceModel


# logging
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="conf", config_name='config')
def main(args: DictConfig) -> None:
    logging.info("Loading models for multi-turn dialogue for test set...")
    random.seed(1)
    
    
    # Load qa_model
    is_vllm = "vllm" in args.qa_model.model_type.lower()
    if not (is_vllm):
        logging.info("Model type not yet supported.")
        return -1
    elif is_vllm:
        qa_model = VLLMInferenceModel(**args.qa_model.model_config)
        BOS_TOKEN, EOS_TOKEN, B_INST, E_INST = '<s>', '</s>', '[INST]', '[/INST]'
    

    # Load human_model
    is_vllm = "vllm" in args.human_model.model_type.lower()
    if not (is_vllm):
        logging.info("Model type not yet supported.")
        return -1
    if is_vllm:
        if args.human_model.name == args.qa_model.name:
        # can't load two mixtral instances.. assume that if is_vllm, then qa_model is also mixtral
        # just point to qa_model
            human_model = qa_model
        else:
            human_model = VLLMInferenceModel(**args.human_model.model_config)
        
    
    # Load prompts
    with open(PROMPTS_DIR, "r") as f:
        prompts = json.load(f)
        prompts = [s.strip() for s in prompts]
    
    
    
    # Load personas
    with open(PERSONAS_DIR, 'r') as f:
        personas = json.load(f)
    # Load names
    with open(NAMES_DIR, 'r') as f:
        names = json.load(f)
        

    final_conversations = defaultdict(list)
    with open(f"simulated-conversations/private/m1_qa-{QA_PROMPT_IDX}_humansys-{HUMAN_SYS_PROMPT_IDX}_human-{HUMAN_PROMPT_IDX}_maxturns-{MAX_TURNS}.json", 'w') as f:
        json.dump(final_conversations, f)
    for j, persona in enumerate(personas):
        logging.info(f"Beginning simulations for persona {j}...")
        prompt_batches = list(batch_list(prompts, args.qa_model.run.batch_size))  # Create batches of prompts
        for batch_index, prompt_batch in enumerate(prompt_batches):
            logging.info(f"Beginning simulations for persona {j}, prompt batch {batch_index}...")
            # Sample number of turns uniformly from integers [1, MAX_TURNS] inclusive
            total_turns = random.randrange(1, MAX_TURNS)
            initial_prompts = [f"{BOS_TOKEN}{B_INST}{QA_PROMPTS[QA_PROMPT_IDX].format(names[j], prompt)}{E_INST}" for prompt in prompt_batch]  
            # TODO: Decide the better approach; another simpler way to do this could have been to just duplicate initial_prompts as necessary here, as we do for each flatten_list call in i.e. what is used in the line below, but some people might find existing solution more elegant
            # initial_prompts = flatten_list([[initial_prompt] * args.qa_model.run.initial_completion_config.num_return_sequences for initial_prompt in initial_prompts])
            # Note that changing the solution here will require a change to all the chunking logic later on
            
            
            # QA Model initial turn
            qa_responses = qa_model.batch_prompt(initial_prompts, **args.qa_model.run.initial_completion_config)  # has length num_return_sequences * batch_size
            conversations = list()
            for i, sublist in enumerate(chunk_list(qa_responses, args.qa_model.run.initial_completion_config.num_return_sequences)):
                conversations.extend([initial_prompts[i] + '\n' + qa_response + EOS_TOKEN for qa_response in sublist])
            
            
            # Human Model initial turn
            conversation_histories = [extract_history(conversation) for conversation in conversations]
            if is_vllm:
                roleplay_prompts = [f"{BOS_TOKEN}{B_INST} {HUMAN_SYS_MSGS[HUMAN_SYS_PROMPT_IDX]}\n\n{HUMAN_PROMPTS[HUMAN_PROMPT_IDX].format(persona, prompt, history) }{E_INST}" for prompt, history in zip(flatten_list([[prompt_batch_item] * args.qa_model.run.initial_completion_config.num_return_sequences for prompt_batch_item in prompt_batch]), conversation_histories)]
                human_responses = human_model.batch_prompt(roleplay_prompts, **args.human_model.run.completion_config)
            conversations = [unfinished_conversation + '\n' + B_INST + f" {human_response} " + E_INST for unfinished_conversation, human_response in zip(conversations, human_responses)]
            curr_turns = 1
            
            
            while curr_turns < total_turns:       
                # QA Model turn
                qa_responses = qa_model.batch_prompt(conversations, **args.qa_model.run.completion_config)
                conversations = [unfinished_conversation + '\n' + qa_response + EOS_TOKEN for unfinished_conversation, qa_response in zip(conversations, qa_responses)]
                
                
                # Human Model turn
                conversation_histories = [extract_history(conversation) for conversation in conversations]
                if is_vllm:
                    roleplay_prompts = [f"{BOS_TOKEN}{B_INST} {HUMAN_SYS_MSGS[HUMAN_SYS_PROMPT_IDX]}\n\n{HUMAN_PROMPTS[HUMAN_PROMPT_IDX].format(persona, prompt, history) }{E_INST}" for prompt, history in zip(flatten_list([[prompt_batch_item] * args.qa_model.run.initial_completion_config.num_return_sequences for prompt_batch_item in prompt_batch]), conversation_histories)]
                    human_responses = human_model.batch_prompt(roleplay_prompts, **args.human_model.run.completion_config)
                
                
                conversations = [unfinished_conversation + '\n' + B_INST + f" {human_response} " + E_INST for unfinished_conversation, human_response in zip(conversations, human_responses)]
                curr_turns += 1
            
            
            conversations = chunk_list(conversations, args.qa_model.run.initial_completion_config.num_return_sequences)
            # Correctly indexing and storing responses for each prompt in the batch
            for i, sublist in enumerate(conversations):
                prompt_index = batch_index * args.qa_model.run.batch_size + i
                pair_key = f"prompt-{prompt_index} persona-{j}"
                final_conversations[pair_key].extend(sublist)
            
            with open(f"simulated-conversations/private/m1_qa-{QA_PROMPT_IDX}_humansys-{HUMAN_SYS_PROMPT_IDX}_human-{HUMAN_PROMPT_IDX}_maxturns-{MAX_TURNS}.json", 'w') as f:
                json.dump(final_conversations, f)
    
    with open(f"simulated-conversations/private/m1_qa-{QA_PROMPT_IDX}_humansys-{HUMAN_SYS_PROMPT_IDX}_human-{HUMAN_PROMPT_IDX}_maxturns-{MAX_TURNS}.json", 'w') as f:
        json.dump(final_conversations, f)


if __name__ == '__main__':
    try:
        fire.Fire(main())
    except:
        pass