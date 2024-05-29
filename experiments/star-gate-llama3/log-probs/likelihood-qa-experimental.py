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
import gc
import time
import os
import signal
from collections import defaultdict
from datasets import load_dataset, Dataset

from utils import *
from paths import *
from transformers import AutoTokenizer

# import models
from AG.models.huggingface.hf_inference_model import HFInferenceModel
from AG.models.openai.azure import AsyncAzureChatLLM
from AG.models.openai.gpt4 import GPT4Agent
from AG.models.vllm_models.inference_model import VLLMInferenceModel


# logging
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="conf", config_name='config')
def main(args: DictConfig) -> None:
    logging.info(f"Loading model for {args.condition.name} group likelihood generation...")
    random.seed(1)
    
    
    # # Load model
    is_vllm = "vllm" in args.model.model_type.lower()
    if not (is_vllm):
        logging.info("Model type not yet supported.")
        return -1
    if is_vllm:
        model = VLLMInferenceModel(**args.model.model_config)
   
    
    # Load prompts
    with open(f"{PROMPT_PATH}/{LLAMA_VERSION}/{args.split.name}.json", "r") as f:
        prompts = json.load(f)
        prompts = [s.strip() for s in prompts]
    
    # Load personas
    with open(f"{PERSONAS_PATH}/{LLAMA_VERSION}/{args.split.name}.json", 'r') as f:
        personas = json.load(f)
    
    # Load names
    with open(f"{PERSONAS_PATH}/{LLAMA_VERSION}/{args.split.name}_NAMES.json", 'r') as f:
        names = json.load(f)
        
    # Load conversations
    with open(f'{SIMULATION_PATH}/{LLAMA_VERSION}/{args.qa_model.shortname}/{args.split.name}-pooled.json', 'r') as f:
        all_conversations = json.load(f)
    
    # Load gold answers
    with open(f'{GOLD_PATH}/{LLAMA_VERSION}/{args.split.name}.json', 'r') as f:
        gold_responses = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(**args.model.tokenizer_config)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    
    if not os.path.exists(f"{LOGPROBS_PATH}/{LLAMA_VERSION}/{args.condition.name}/{args.qa_model.shortname}"):
        os.makedirs(f"{LOGPROBS_PATH}/{LLAMA_VERSION}/{args.condition.name}/{args.qa_model.shortname}")
    if torch.cuda.is_available():
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No GPU available, using CPU")
    final_log_probs = defaultdict(list)
    for j, persona in enumerate(personas):
        for i, prompt in enumerate(prompts):
            if i % 5 == 0:
                logging.info(f'CHECKPOINT: Computing batched log probabilities for prompt-{i} persona-{j}...')
                print(f'CHECKPOINT: Computing batched log probabilities for prompt-{i} persona-{j}...')
            conversations = all_conversations[f"prompt-{i} persona-{j}"]
            # split conversations in half not to overload GPU memory
            #first, second, third = conversations[:len(conversations)//3], conversations[len(conversations)//3:2*len(conversations)//3], conversations[2*len(conversations)//3:]

            if is_vllm:
                for lst_idx, lst in enumerate([[c] for c in conversations]):
                    if len(lst) == 0: continue
                    try:
                        jsonprompts=[[{"role" : "user", "content" : f"My name is {names[j].strip()}.\n\n{prompts[i].strip()}"}] + conversation[1:] for conversation in lst]
                        final_turns = [jsonlist[-1]['content'].strip() + '\n\n' + prompt.strip() for jsonlist in jsonprompts]
                        
                        prechatified_prompts = [jsonlist[:-1] + [{"role" : "user", "content" : final_turns[curr]}] for curr, jsonlist in enumerate(jsonprompts)]
                        postchatified_prompts = [tokenizer.apply_chat_template(jsonlist, tokenize=False) for jsonlist in prechatified_prompts]
                        answers = [jsonlist + [{"role": "assistant", "content": gold_responses[f'prompt-{i} persona-{j}'][0].strip()}] for jsonlist in prechatified_prompts]
                        
                        answers = [tokenizer.apply_chat_template(jsonlist, tokenize=False) for jsonlist in answers]
                        log_probs = model.batch_log_probs(
                            prompts=postchatified_prompts,
                            responses=answers
                        ).to('cpu')
                        final_log_probs[f'prompt-{i} persona-{j}'].extend(log_probs.tolist())
                        del log_probs
                        torch.cuda.empty_cache()
                        gc.collect()  # Explicitly invoke garbage collection
                    except RuntimeError as e:
                        print("OOM error, skipping batch ", lst_idx, " for prompt ", i, " persona ", j)
                        print(e)
                        final_log_probs[f'prompt-{i} persona-{j}'].extend([float('-inf') for _ in range(len(lst))])
                        torch.cuda.empty_cache()
                        gc.collect()  # Explicitly invoke garbage collection
        with open(f"{LOGPROBS_PATH}/{LLAMA_VERSION}/{args.condition.name}/{args.qa_model.shortname}/{args.split.name}.json", 'w') as f:
            json.dump(final_log_probs, f)
                    


    if not os.path.exists(f"{LOGPROBS_PATH}/{LLAMA_VERSION}/{args.condition.name}/{args.qa_model.shortname}"):
        os.makedirs(f"{LOGPROBS_PATH}/{LLAMA_VERSION}/{args.condition.name}/{args.qa_model.shortname}")
    with open(f"{LOGPROBS_PATH}/{LLAMA_VERSION}/{args.condition.name}/{args.qa_model.shortname}/{args.split.name}.json", 'w') as f:
        json.dump(final_log_probs, f)
        

if __name__ == '__main__':
    try:
        fire.Fire(main())
    except:
        pass
