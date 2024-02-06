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

from utils import GENERATION_PROMPTS, GENERATION_PROMPT_IDX, SYS_PROMPTS, SYS_PROMPT_IDX, PERSONAS_DIR, PROMPTS_DIR, flatten_list

# import models
from AG.models.huggingface.hf_inference_model import HFInferenceModel
from AG.models.openai.azure import AsyncAzureChatLLM
from AG.models.openai.gpt4 import GPT4Agent
from AG.models.vllm_models.inference_model import VLLMInferenceModel


# logging
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="conf", config_name='config')
def main(args: DictConfig) -> None:
    logging.info("Loading model for gold QA generation...")
    random.seed(1)
    
    # Load model
    is_openai = "openai" in args.model.model_type.lower()
    if not (is_openai):
        logging.info("Model type not yet supported.")
        return -1
    if is_openai:
        args.model.model_config.azure_api.api_key = os.getenv("OPENAI_API_KEY")
        llm = AsyncAzureChatLLM(**args.model.model_config.azure_api)
        model = GPT4Agent(llm=llm, **args.model.run.completion_config)
    
    
    
        
    
    # Load prompts
    with open(PROMPTS_DIR, "r") as f:
        prompts = json.load(f)
        prompts = [s.strip() for s in prompts]
        
    
    # Load personas
    with open(PERSONAS_DIR, 'r') as f:
        personas = json.load(f)
        
        
    gold_responses = defaultdict(list)
    
    for j, persona in enumerate(personas):
        for i, prompt in enumerate(prompts):
            responses = model.batch_prompt(system_message=SYS_PROMPTS[SYS_PROMPT_IDX], messages=[GENERATION_PROMPTS[GENERATION_PROMPT_IDX].format(persona, prompt)],)
            responses = flatten_list(responses)

            gold_responses[f"prompt-{i} persona-{j}"].extend(responses)
    
    with open(f"gold-responses/model-{args.model.name}_gen-{GENERATION_PROMPT_IDX}_sys-{SYS_PROMPT_IDX}_temp-{args.model.run.completion_config.temperature}_n-{args.model.run.completion_config.n}.json", 'w') as f:
        json.dump(gold_responses, f)
            
    
if __name__ == '__main__':
    try:
        fire.Fire(main())
    except:
        pass