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

from utils import QA_PROMPTS, QA_PROMPT_IDX, HUMAN_PROMPTS, HUMAN_PROMPT_IDX, HUMAN_SYS_MSGS, HUMAN_SYS_PROMPT_IDX, NAMES_DIR, PERSONAS_DIR, PROMPTS_DIR, MAX_TURNS, flatten_list, batch_list, chunk_list

# import models
from AG.models.huggingface.hf_inference_model import HFInferenceModel
from AG.models.openai.azure import AsyncAzureChatLLM
from AG.models.openai.gpt4 import GPT4Agent
from AG.models.vllm_models.inference_model import VLLMInferenceModel


# logging
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="conf", config_name='config')
def main(args: DictConfig) -> None:
    logging.info("Loading models for multi-turn dialogue...")
    random.seed(1)
    
    
    # Load qa_model
    is_hf = "hf" in args.qa_model.model_type.lower()
    is_vllm = "vllm" in args.qa_model.model_type.lower()
    if not (is_hf or is_vllm):
        logging.info("Model type not yet supported.")
        return -1
    
    if is_hf:
        qa_model = HFInferenceModel(**args.qa_model.model_config)
    elif is_vllm:
        qa_model = VLLMInferenceModel(**args.qa_model.model_config)
        #q a_model = 
    
    
    # Load human_model
    is_openai = "openai" in args.human_model.model_type.lower()
    is_vllm = "vllm" in args.human_model.model_type.lower()
    if not (is_openai or is_vllm):
        logging.info("Model type not yet supported.")
        return -1
    
    if is_openai:
        args.human_model.model_config.azure_api.api_key = os.getenv("OPENAI_API_KEY")
        human_llm = AsyncAzureChatLLM(**args.human_model.model_config.azure_api)
        human_model = GPT4Agent(llm=human_llm, **args.human_model.run.completion_config)
    elif is_vllm:
        # can't load two mixtral instances.. assume that if is_vllm, then qa_model is also mixtral
        # just point to qa_model
        human_model = qa_model
        
    
    # Load prompts
    with open(PROMPTS_DIR, "r") as f:
        prompts = json.load(f)
        prompts = [s.strip() for s in prompts]
    BOS_TOKEN, EOS_TOKEN, B_INST, E_INST = '<s>', '</s>', '[INST]', '[/INST]'
    
    # Load personas
    with open(PERSONAS_DIR, 'r') as f:
        personas = json.load(f)
    # Load names
    with open(NAMES_DIR, 'r') as f:
        names = json.load(f)
        

    final_conversations = defaultdict(list)
    
    
            
            
    for j, persona in enumerate(personas):
        logging.info(f"Beginning simulations for persona {j}...")
        prompt_batches = list(batch_list(prompts, args.qa_model.run.batch_size))  # Create batches of prompts
        for batch_index, prompt_batch in enumerate(prompt_batches):
            logging.info(f"Beginning simulations for persona {j}, prompt batch {batch_index}...")
            # Sample number of turns uniformly from integers [1, MAX_TURNS] inclusive
            total_turns = random.randrange(1, MAX_TURNS)
            total_turns = 2
            
            
            initial_prompts = [f"{BOS_TOKEN}{B_INST}{QA_PROMPTS[QA_PROMPT_IDX].format(names[j], prompt)}{E_INST}" for prompt in prompt_batch]  
            # TODO: Decide the better approach; another simpler way to do this could have been to just duplicate initial_prompts as necessary here, as we do for each flatten_list call in i.e. what is used in the line below, but some people might find existing solution more elegant
            # initial_prompts = flatten_list([[initial_prompt] * args.qa_model.run.initial_completion_config.num_return_sequences for initial_prompt in initial_prompts])
            # Note that changing the solution here will require a change to all the chunking logic later on
            
            
            qa_responses = qa_model.batch_prompt(initial_prompts, **args.qa_model.run.initial_completion_config)  # has length num_return_sequences * batch_size
            conversations = list()
            for i, sublist in enumerate(chunk_list(qa_responses, args.qa_model.run.initial_completion_config.num_return_sequences)):
                conversations.extend([initial_prompts[i] + '\n' + qa_response + EOS_TOKEN for qa_response in sublist])
                
                
            if is_openai:
                roleplay_prompts = [HUMAN_PROMPTS[HUMAN_PROMPT_IDX].format(persona, prompt, qa_response) for prompt, qa_response in zip(flatten_list([[initial_prompt] * args.qa_model.run.initial_completion_config.num_return_sequences for initial_prompt in initial_prompts]), qa_responses)]
                human_responses = human_model.batch_prompt(system_message=HUMAN_SYS_MSGS[HUMAN_SYS_PROMPT_IDX], messages=roleplay_prompts,)
                human_responses = flatten_list(human_responses)
            elif is_vllm:
                roleplay_prompts = [f"{BOS_TOKEN}{B_INST} {HUMAN_SYS_MSGS[HUMAN_SYS_PROMPT_IDX]}\n\n{HUMAN_PROMPTS[HUMAN_PROMPT_IDX].format(persona, prompt, qa_response) }{E_INST}" for prompt, qa_response in zip(flatten_list([[initial_prompt] * args.qa_model.run.initial_completion_config.num_return_sequences for initial_prompt in initial_prompts]), qa_responses)]
                human_responses = human_model.batch_prompt(roleplay_prompts, **args.human_model.run.completion_config)
            conversations = [unfinished_conversation + '\n' + B_INST + f" A: {human_response} " + E_INST for unfinished_conversation, human_response in zip(conversations, human_responses)]
            curr_turns = 1
            
            
            while curr_turns < total_turns:       
                qa_responses = qa_model.batch_prompt(conversations, **args.qa_model.run.completion_config)
                conversations = [unfinished_conversation + '\n' + qa_response + EOS_TOKEN for unfinished_conversation, qa_response in zip(conversations, qa_responses)]
                if is_openai:
                    roleplay_prompts = [HUMAN_PROMPTS[HUMAN_PROMPT_IDX].format(persona, prompt, qa_response) for prompt, qa_response in zip(flatten_list([[initial_prompt] * args.qa_model.run.initial_completion_config.num_return_sequences for initial_prompt in initial_prompts]), qa_responses)]
                    human_responses = human_model.batch_prompt(system_message=HUMAN_SYS_MSGS[HUMAN_SYS_PROMPT_IDX], messages=roleplay_prompts,)
                    human_responses = flatten_list(human_responses)
                elif is_vllm:
                    roleplay_prompts = [f"{BOS_TOKEN}{B_INST} {HUMAN_SYS_MSGS[HUMAN_SYS_PROMPT_IDX]}\n\n{HUMAN_PROMPTS[HUMAN_PROMPT_IDX].format(persona, prompt, qa_response) }{E_INST}" for prompt, qa_response in zip(flatten_list([[initial_prompt] * args.qa_model.run.initial_completion_config.num_return_sequences for initial_prompt in initial_prompts]), qa_responses)]
                    human_responses = human_model.batch_prompt(roleplay_prompts, **args.human_model.run.completion_config)
                conversations = [unfinished_conversation + '\n' + B_INST + f" A: {human_response} " + E_INST for unfinished_conversation, human_response in zip(conversations, human_responses)]
                curr_turns += 1
            
            
            conversations = chunk_list(conversations, args.qa_model.run.initial_completion_config.num_return_sequences)
            # Correctly indexing and storing responses for each prompt in the batch
            for i, sublist in enumerate(conversations):
                prompt_index = batch_index * args.qa_model.run.batch_size + i
                pair_key = f"prompt-{prompt_index} persona-{j}"
                final_conversations[pair_key].extend(sublist)
            
            
            final_conversations[f"prompt-{i} persona-{j}"] = conversations
            with open(f"simulated-conversations/qa-model-{args.qa_model.name}_human-model-{args.human_model.name}_qa-{QA_PROMPT_IDX}_humansys-{HUMAN_SYS_PROMPT_IDX}_human-{HUMAN_PROMPT_IDX}_maxturns-{MAX_TURNS}.json", 'w') as f:
                json.dump(final_conversations, f)
    
    with open(f"simulated-conversations/qa-model-{args.qa_model.name}_human-model-{args.human_model.name}_qa-{QA_PROMPT_IDX}_humansys-{HUMAN_SYS_PROMPT_IDX}_human-{HUMAN_PROMPT_IDX}_maxturns-{MAX_TURNS}.json", 'w') as f:
        json.dump(final_conversations, f)


if __name__ == '__main__':
    try:
        fire.Fire(main())
    except:
        pass