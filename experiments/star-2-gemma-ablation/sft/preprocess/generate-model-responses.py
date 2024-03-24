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
from transformers import AutoModelForCausalLM, AutoTokenizer

# import models
from AG.models.huggingface.hf_inference_model import HFInferenceModel
from AG.models.openai.azure import AsyncAzureChatLLM
from AG.models.openai.gpt4 import GPT4Agent
from AG.models.vllm_models.inference_model import VLLMInferenceModel
from paths import *

from utils import *


# logging
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="conf", config_name='config')
def main(args: DictConfig) -> None:
    logging.info("Loading raw simulated conversations for preprocessing...")
    qa_model = VLLMInferenceModel(**args.qa_model.model_config)
    tokenizer = AutoTokenizer.from_pretrained(**args.qa_model.tokenizer_config)
    
    
    with open(f"{SIMULATION_PATH}/{VERSION_2_MISTRAL_ABLATION}/{args.qa_model.shortname}/{args.split.name}_top-k-{args.k}.json", 'r') as f:
        conversations = json.load(f)
    with open(f'{PERSONAS_PATH}/{VERSION_2_MISTRAL_ABLATION}/{args.split.name}.json', "r") as f:
        personas = json.load(f)
    with open(f'{PERSONAS_PATH}/{VERSION_2_MISTRAL_ABLATION}/{args.split.name}_NAMES.json', "r") as f:
        names = json.load(f)

    
    output_responses = dict()
    for j, persona in enumerate(personas):
        logging.info(f"Beginning simulations for persona {j}...")
        
        persona_keys = [key for key in conversations.keys() if key.strip().endswith(f'persona-{j}')]
        initial_persona_conversations = [extract_history(c) for c in flatten_list([conversations[key] for key in persona_keys])]
        
        persona_conversations = list()
        for ipc in initial_persona_conversations:
            turns = create_turns(ipc)
            
            # These two lines specifically for SFT target including gold response
            turns[-1] = turns[-1].strip() + f'\n\n{turns[0].strip()}'  ## add the prompt to the end of the conversation again to prompt the model to answer
            turns[0] = MODEL_RESPONSE_PROMPTS[args.MODEL_RESPONSE_IDX] + '\n\n' + f"My name is {names[j].strip()}.\n\n{turns[0].strip()}"
            messages = [{"role": args.ROLES[i % 2], "content": turn} for i, turn in enumerate(turns)]
            persona_conversations.append(tokenizer.decode(tokenizer.apply_chat_template(messages)))

        #persona_conversations = [BOS_TOKEN + B_INST + MODEL_RESPONSE_PROMPTS[args.MODEL_RESPONSE_IDX] + extract_history(c) for c in flatten_list(persona_conversations)]
        conversation_batches = list(batch_list(persona_conversations, args.qa_model.run.batch_size))  # Create batches of prompts        
        final_qa_responses = list()
        for batch_index, conversation_batch in enumerate(conversation_batches):
            logging.info(f"Running batch {batch_index} of {len(conversation_batches)}...")
            qa_responses = qa_model.batch_prompt(conversation_batch, **args.qa_model.run.completion_config)
            final_qa_responses.extend(qa_responses)

        
        output_responses = output_responses | dict(zip(persona_keys, [[res] for res in final_qa_responses]))
            
    if not os.path.exists(f"{MODELRESPONSE_PATH}/{VERSION_2_MISTRAL_ABLATION}/{args.qa_model.shortname}"):
        os.makedirs(f"{MODELRESPONSE_PATH}/{VERSION_2_MISTRAL_ABLATION}/{args.qa_model.shortname}")
    
    with open(f"{MODELRESPONSE_PATH}/{VERSION_2_MISTRAL_ABLATION}/{args.qa_model.shortname}/{args.split.name}_top-k-{args.k}.json", 'w') as f:
        json.dump(dict(output_responses), f)
    
    return -1


if __name__ == '__main__':
    try:
        fire.Fire(main())
    except:
        pass


