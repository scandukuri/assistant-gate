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
import copy
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
    
    
    with open(f"{SIMULATION_PATH}/{LLAMA_VERSION}/{args.qa_model.shortname}/{args.split.name}_top-k-{args.k}.json", 'r') as f:
        conversations = json.load(f)
    with open(f'{PERSONAS_PATH}/{LLAMA_VERSION}/{args.split.name}.json', "r") as f:
        personas = json.load(f)
    with open(f'{PERSONAS_PATH}/{LLAMA_VERSION}/{args.split.name}_NAMES.json', "r") as f:
        names = json.load(f)
    # Load prompts
    with open(f'{PROMPT_PATH}/{LLAMA_VERSION}/{args.split.name}.json', "r") as f:
        prompts = json.load(f)
        prompts = [s.strip() for s in prompts]
    keys = list(conversations.keys())
    conversations = [conversations[key] for key in keys]
    output_conversations = defaultdict(list)
    
    prompt_indices = get_prompt_keys(keys)
    persona_indices = get_persona_keys(keys)


    # Prepare data for processing
    flat_conversations = flatten_list(conversations)
    flat_prompt_ids = flatten_list([[prompt_id] * args.qa_model.run.completion_config.num_return_sequences for prompt_id in prompt_indices])
    flat_persona_ids = flatten_list([[persona_id] * args.qa_model.run.completion_config.num_return_sequences for persona_id in persona_indices])
    flat_personas = [personas[persona_id] for persona_id in flat_persona_ids]
    flat_prompts = [prompts[prompt_id] for prompt_id in flat_prompt_ids]
    
    
    # Generate batches for processing
    prompt_batches = list(batch_list(flat_prompts, args.qa_model.run.batch_size))
    persona_batches = list(batch_list(flat_personas, args.qa_model.run.batch_size))
    conversation_batches = list(batch_list(flat_conversations, args.qa_model.run.batch_size))
    
    
    # Generate responses
    final_conversations = []
    for batch_index, (prompt_batch, persona_batch, conversation_batch) in enumerate(zip(prompt_batches, persona_batches, conversation_batches)):
        logging.info(f"Running batch {batch_index + 1} of {len(conversation_batches)}...")
        answerer_prompts = copy.deepcopy(conversation_batch)
        for i, answerer_prompt in enumerate(answerer_prompts):
            answerer_prompt[-1]["content"] = answerer_prompt[-1]["content"] + '\n\n' + prompt_batch[i]
            answerer_prompt[0]["content"] = MODEL_RESPONSE_PROMPTS[args.MODEL_RESPONSE_IDX] + '\n\n' + prompt_batch[i]
        answers = qa_model.batch_prompt([tokenizer.apply_chat_template(answerer_prompt, tokenize=False) for answerer_prompt in answerer_prompts], **args.qa_model.run.completion_config)
        appended_conversations = [c + [{"role" : "assistant", "content" : r.strip()}] for c, r in zip(answerer_prompts, answers)]
        final_conversations.extend(appended_conversations)
        
        
    # Construct output conversations
    for flat_index, conversation in enumerate(final_conversations):
        prompt_index, persona_index = flat_prompt_ids[flat_index], flat_persona_ids[flat_index]
        output_conversations[f"prompt-{prompt_index} persona-{persona_index}"].append(conversation)
            
    if not os.path.exists(f"{MODELRESPONSE_PATH}/{LLAMA_VERSION}/{args.qa_model.shortname}"):
        os.makedirs(f"{MODELRESPONSE_PATH}/{LLAMA_VERSION}/{args.qa_model.shortname}")
    
    with open(f"{MODELRESPONSE_PATH}/{LLAMA_VERSION}/{args.qa_model.shortname}/{args.split.name}_top-k-{args.k}.json", 'w') as f:
        json.dump(dict(output_conversations), f)
    
    return -1


if __name__ == '__main__':
    try:
        fire.Fire(main())
    except:
        pass


