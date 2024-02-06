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

from utils import QA_PROMPTS, QA_PROMPT_IDX, HUMAN_PROMPTS, HUMAN_PROMPT_IDX, HUMAN_SYS_MSGS, HUMAN_SYS_PROMPT_IDX, PERSONAS_DIR, PROMPTS_DIR, MAX_TURNS, filter_completed_conversations, flatten_list

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
    random.seed(43728)
    
    
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
        
        

    final_conversations = defaultdict(list)
    for i, prompt in enumerate(prompts):
        if i % 10 == 0:
            with open(f"simulated-conversations/qa-model-{args.qa_model.name}_human-model-{args.human_model.name}_qa-{QA_PROMPT_IDX}_humansys-{HUMAN_SYS_PROMPT_IDX}_human-{HUMAN_PROMPT_IDX}_maxturns-{MAX_TURNS}.json", 'w') as f:
                json.dump(final_conversations, f)
        for j, persona in enumerate(personas):
            completed_conversations = []
            initial_prompt = f"{BOS_TOKEN}{B_INST} {QA_PROMPTS[QA_PROMPT_IDX]}\n\n{persona}\n\n{prompt} {E_INST}"
            qa_responses = qa_model.batch_prompt([initial_prompt], **args.qa_model.run.initial_completion_config)
            
            unfinished_conversations = [initial_prompt + '\n' + qa_response + EOS_TOKEN for qa_response in qa_responses]
            unfinished_conversations, newly_completed_conversations = filter_completed_conversations(QA_PROMPTS[QA_PROMPT_IDX], unfinished_conversations)
            completed_conversations.extend(newly_completed_conversations)
            turns = 1
            
            while len(unfinished_conversations) > 0 and turns < MAX_TURNS:
                if is_openai:
                    human_responses = human_model.batch_prompt(system_message=HUMAN_SYS_MSGS[HUMAN_SYS_PROMPT_IDX], messages=[HUMAN_PROMPTS[HUMAN_PROMPT_IDX].format(persona, prompt, qa_response[2:]) for qa_response in qa_responses],)
                    human_responses = flatten_list(human_responses)
                elif is_vllm:
                    human_responses = human_model.batch_prompt([f"{BOS_TOKEN}{B_INST} {HUMAN_SYS_MSGS[HUMAN_SYS_PROMPT_IDX]}\n\n{HUMAN_PROMPTS[HUMAN_PROMPT_IDX].format(persona, prompt, qa_response[2:]) }{E_INST}" for qa_response in qa_responses], **args.human_model.run.completion_config)
                unfinished_conversations = [unfinished_conversation + '\n' + B_INST + f" A: {human_response} " + E_INST for unfinished_conversation, human_response in zip(unfinished_conversations, human_responses)]


                qa_responses = qa_model.batch_prompt(unfinished_conversations, **args.qa_model.run.completion_config)
                unfinished_conversations = [unfinished_conversation + '\n' + qa_response + EOS_TOKEN for unfinished_conversation, qa_response in zip(unfinished_conversations, qa_responses)]

                
                unfinished_conversations, newly_completed_conversations = filter_completed_conversations(QA_PROMPTS[QA_PROMPT_IDX], unfinished_conversations)
                completed_conversations.extend(newly_completed_conversations)
                turns += 1

            final_conversations[f"prompt-{i} persona-{j}"] = completed_conversations
    
    with open(f"simulated-conversations/qa-model-{args.qa_model.name}_human-model-{args.human_model.name}_qa-{QA_PROMPT_IDX}_humansys-{HUMAN_SYS_PROMPT_IDX}_human-{HUMAN_PROMPT_IDX}_maxturns-{MAX_TURNS}.json", 'w') as f:
        json.dump(final_conversations, f)


if __name__ == '__main__':
    try:
        fire.Fire(main())
    except:
        pass