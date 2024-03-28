import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import json
from utils import *
import hydra
from omegaconf import DictConfig
import argparse
import fire
import pandas as pd
from tqdm import tqdm
import numpy as np
import json
import logging
import torch
import random
import logging
import re
import time
import os
import signal
from collections import defaultdict
from datasets import load_dataset, Dataset
import requests


from utils import *
from paths import *


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
    random.seed(1)
    
    random.seed(1)
    
    rating_llm = AsyncAzureChatLLM(**args.rating_model.model_config.azure_api)
    rating_model = GPT4Agent(llm=rating_llm, **args.rating_model.run.completion_config)
    
    with open(f"{PROMPT_PATH}/{VERSION_2_BSFT}/{args.split.name}.json", "r") as f:
        prompts = json.load(f)
        prompts = [s.strip() for s in prompts]
    
    output_ratings = list()
    for prompt in prompts:
        rating_messages = rating_model.batch_prompt(system_message=RATING_SYS_PROMPTS[args.RATING_SYS_PROMPT_IDX], messages=[RATING_PROMPTS[args.RATING_PROMPT_IDX].format(prompt)])
        output_ratings.extend(rating_messages[0])
    
    final_ratings = list()
    for message in output_ratings:
        try:
            final_ratings.append(int(message[message.find('Final Response:') + len('Final Response:'):].strip()))
        except:
            final_ratings.append(0)

    if not os.path.exists(f"{SPECIFICITY_PATH}/{VERSION_2_BSFT}"):
        os.makedirs(f"{SPECIFICITY_PATH}/{VERSION_2_BSFT}")
    with open(f"{SPECIFICITY_PATH}/{VERSION_2_BSFT}/{args.split.name}_ratings.json", 'w') as f:
        json.dump(final_ratings, f)
    logging.info(f"Final ratings...")
    print(final_ratings)

if __name__ == '__main__':
    try:
        fire.Fire(main())
    except:
        pass
