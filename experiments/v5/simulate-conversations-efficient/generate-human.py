
#####
# TODO: THIS WHOLE SCRIPT
#####




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
    
    
    pulled_conversations = json.load(open(f"simulated-conversations/{args.qa_model.shortname}_{args.split.name}.json", 'r'))
    output_conversations = defaultdict(list)
    human_model = VLLMInferenceModel(**args.human_model.model_config)
    for j, persona in enumerate(personas):
        logging.info(f"Beginning simulations for persona {j}...")
        
        # get list of all prompts for that persona in order
        conversations = [pulled_conversations[f'prompt-{p} persona-{j}'] for p in range(len(prompts))]

        flattened_prompts = flatten_list([[prompt] * args.qa_model.run.initial_completion_config.num_return_sequences for prompt in prompts])

        flattened_conversations = flatten_list(conversations)

        prompt_batches = list(batch_list(flattened_prompts, args.qa_model.run.batch_size))

        conversation_batches = list(batch_list(flattened_conversations, args.qa_model.run.batch_size))  # Create batches of prompts

        
        
        final_conversations = list()

        for batch_index, conversation_batch in enumerate(conversation_batches):
            histories = [extract_history(c) for c in conversation_batch]

            roleplay_prompts = [f"{BOS_TOKEN}{B_INST} {HUMAN_SYS_MSGS[args.HUMAN_SYS_PROMPT_IDX]}\n\n{HUMAN_PROMPTS[args.HUMAN_PROMPT_IDX].format(persona, prompt, history) }{E_INST}" for prompt, history in zip(prompt_batches[batch_index], histories)]

            human_responses = human_model.batch_prompt(roleplay_prompts, **args.human_model.run.completion_config)

            appended_conversations = [unfinished_conversation + '\n' + B_INST + f" {human_response} " + E_INST for unfinished_conversation, human_response in zip(conversation_batch, human_responses)]

            final_conversations.extend(appended_conversations)
            
        
        final_conversations = chunk_list(final_conversations, args.qa_model.run.initial_completion_config.num_return_sequences)
        for i, sublist in enumerate(final_conversations):
            pair_key = f"prompt-{i} persona-{j}"
            output_conversations[pair_key].extend(sublist)
            
    
    with open(f"simulated-conversations/{args.qa_model.shortname}_{args.split.name}.json", 'w') as f:
        json.dump(output_conversations, f)


if __name__ == '__main__':
    try:
        fire.Fire(main())
    except:
        pass
