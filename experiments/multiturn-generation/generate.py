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

from utils import GENERATION_PROMPTS, PROMPT_IDX, PERSONAS_DIR

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
    
    
    # Load qa_model
    is_hf = "hf" in args.qa_model.model_type.lower()
    if not is_hf:
        logging.info("Model type not yet supported.")
        return -1
    qa_model = HFInferenceModel(**args.qa_model.model_config)
    
    
    # Load human_model
    is_openai = "openai" in args.human_model.model_type.lower()
    if not is_openai:
        logging.info("Model type not yet supported.")
        return -1
    args.human_model.model_config.azure_api.api_key = os.getenv("OPENAI_API_KEY")
    human_llm = AsyncAzureChatLLM(**args.human_model.model_config.azure_api)
    human_model = GPT4Agent(llm=human_llm, **args.human_model.run.completion_config)
        
    
    # Load prompts
    with open("instruct-questions/first-50-prompts.json", "r") as f:
        prompts = json.load(f)
        prompts = [s.strip() for s in prompts]
    BOS_TOKEN, EOS_TOKEN, B_INST, E_INST = '<s>', '</s>', '[INST]', '[/INST]'
    
    # Load personas
    with open(PERSONAS_DIR, 'r') as f:
        personas = json.load(f)
        
    test_prompt = f"{BOS_TOKEN}{B_INST} {GENERATION_PROMPTS[PROMPT_IDX]}\n\n{personas[0]}\n\n{prompts[0]} {E_INST}"
    breakpoint()
    completions = qa_model.batch_prompt([test_prompt], **args.qa_model.run.completion_config)
    
    # for prompt in prompts:
    #     for i in range(5):
    #         completions = qa_model.batch_prompt([prompt], **args.qa_model.run.completion_config)
    
        



if __name__ == '__main__':
    try:
        fire.Fire(main())
    except:
        pass