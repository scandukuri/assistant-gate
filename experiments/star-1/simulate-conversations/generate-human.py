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
from paths import *

# logging
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="conf", config_name='config')
def main(args: DictConfig) -> None:
    logging.info(f"Loading models for multi-turn dialogue for {args.split.name}...")
    random.seed(1)
    if args.turn.number > args.MAX_TURNS:
        logging.info(f"Cannot exceed {args.MAX_TURNS} turns per conversation. Exiting.")
        return
    
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
    
    pulled_conversations = json.load(open(f"{SIMULATION_PATH}/{VERSION}/{args.qa_model.shortname}/{args.split.name}.json", 'r'))
    output_conversations = defaultdict(list)
    human_model = VLLMInferenceModel(**args.human_model.model_config)
    for j, persona in enumerate(personas):
        logging.info(f"Beginning simulations for persona {j}...")
        
        # get list of all prompts for that persona in order
        # get all keys from pulled_conversation with 
        raw_conversation_keys = [key if key.strip().endswith(f'persona-{j}') else None for key in pulled_conversations.keys()]
        conversation_keys = list()
        for key in raw_conversation_keys:
            if key is not None:
                conversation_keys.append(key)
        prompt_keys = [int(key[key.find('prompt-') + len('prompt-') : key.find('persona')].strip()) for key in conversation_keys]
        interest_prompts = [prompts[prompt_id] for prompt_id in prompt_keys]
        conversations = [pulled_conversations[f'prompt-{right_key} persona-{j}'] for right_key in prompt_keys] 
        flattened_prompt_ids = flatten_list([[prompt_id] * args.qa_model.run.initial_completion_config.num_return_sequences for prompt_id in prompt_keys])
        flattened_prompts = flatten_list([[prompt] * args.qa_model.run.initial_completion_config.num_return_sequences for prompt in interest_prompts])

        flattened_conversations = flatten_list(conversations)

        prompt_batches = list(batch_list(flattened_prompts, args.human_model.run.batch_size))

        conversation_batches = list(batch_list(flattened_conversations, args.human_model.run.batch_size))  # Create batches of prompts        
        final_conversations = list()

        for batch_index, conversation_batch in enumerate(conversation_batches):
            logging.info(f"Running batch {batch_index} of {len(conversation_batches)}...")
            histories = [extract_history(c) for c in conversation_batch]

            roleplay_prompts = [f"{BOS_TOKEN}{B_INST} {HUMAN_SYS_MSGS[args.HUMAN_SYS_PROMPT_IDX]}\n\n{HUMAN_PROMPTS[args.HUMAN_PROMPT_IDX].format(persona, prompt, history) }{E_INST}" for prompt, history in zip(prompt_batches[batch_index], histories)]

            human_responses = human_model.batch_prompt(roleplay_prompts, **args.human_model.run.completion_config)

            appended_conversations = [unfinished_conversation + '\n' + B_INST + f" {human_response} " + E_INST for unfinished_conversation, human_response in zip(conversation_batch, human_responses)]

            final_conversations.extend(appended_conversations)
            
        
        final_conversations = chunk_list(final_conversations, args.qa_model.run.initial_completion_config.num_return_sequences)
        final_prompt_ids = chunk_list(flattened_prompt_ids, args.qa_model.run.initial_completion_config.num_return_sequences)
        for i, sublist in enumerate(final_conversations):
            pair_key = f"prompt-{prompt_keys[i]} persona-{j}"
            output_conversations[pair_key].extend(sublist)

    if args.turn.number != args.MAX_TURNS:
        # sample int over uniform [int(args.turn.number), args.MAX_TURNS] inclusive
        samples = [random.randint(int(args.turn.number), args.MAX_TURNS) for key in output_conversations.keys()]
        # create mask 1 if int == args.turn.number else 0
        mask = [1 if sample == args.turn.number else 0 for sample in samples]
        finished_keys, unfinished_keys = [key for key, mask_value in zip(output_conversations.keys(), mask) if mask_value == 1], [key for key, mask_value in zip(output_conversations.keys(), mask) if mask_value == 0]
        # Creating finished and unfinished dictionaries from those keys
        finished, unfinished = {key: output_conversations[key] for key in finished_keys}, {key: output_conversations[key] for key in unfinished_keys}
    
        print('Finished: ', finished.keys())
        print('Continuing', unfinished.keys())
        if not os.path.exists(f"{SIMULATION_PATH}/{VERSION}/{args.qa_model.shortname}"):
            os.makedirs(f"{SIMULATION_PATH}/{VERSION}/{args.qa_model.shortname}")

        with open(f"{SIMULATION_PATH}/{VERSION}/{args.qa_model.shortname}/{args.split.name}.json", 'w') as f:
            json.dump(unfinished, f)
        with open(f"{SIMULATION_PATH}/{VERSION}/{args.qa_model.shortname}/{args.split.name}_turn-{args.turn.number}.json", 'w') as f:
            json.dump(finished, f)
    else:
        with open(f"{SIMULATION_PATH}/{VERSION}/{args.qa_model.shortname}/{args.split.name}_turn-{args.turn.number}.json", 'w') as f:
            json.dump(output_conversations, f)
        



if __name__ == '__main__':
    try:
        fire.Fire(main())
    except:
        pass

