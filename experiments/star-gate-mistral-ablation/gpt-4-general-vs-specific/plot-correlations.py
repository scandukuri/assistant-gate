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
from scipy import stats
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

    with open(f"{SPECIFICITY_PATH}/{VERSION_2_MISTRAL_ABLATION}/{args.split.name}_ratings.json", "r") as f:
        ratings = json.load(f)
    with open(f"{LOGPROBS_PATH}/{VERSION_1_ESFT}/qa-experimental/m1/{args.split.name}.json", "r") as f:
        logprobs = json.load(f)
    with open(f"{SIMULATION_PATH}/{VERSION_2_MISTRAL_ABLATION}/m3/{args.split.name}.json", "r") as f:
        conversations = json.load(f)
    breakpoint()
    # Initialize a list of 50 sublists
    split_data = [[None for _ in range(10)] for _ in range(50)]

    # Iterate over the items in the dictionary
    for key, value in logprobs.items():
        # Parse the prompt and persona numbers from the key
        prompt_num, persona_num = map(int, key.replace('prompt-', '').replace('persona-', '').split())

        # Insert the value into the correct position
        split_data[prompt_num][persona_num] = value[0]  # Assuming you want the single value, not the list
    log_probabilities = flatten_list(split_data)
    ratings = flatten_list([[num] * 10 for num in ratings])
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(ratings, log_probabilities)

    # # Create a scatter plot of the ratings vs. log probabilities
    plt.figure(figsize=(10, 6))
    plt.scatter(ratings, log_probabilities, color='blue', alpha=0.5, label='Data points')

    # Add a line of best fit
    line = slope * np.array([1, 2, 3, 4, 5, 6]) + intercept
    plt.plot([1, 2, 3, 4, 5, 6], line, color='red', label='Line of best fit')

    # Enhance the plot
    plt.title('Linear Regression of Ratings vs. Log Probabilities')
    plt.xlabel('Ratings')
    plt.ylabel('Log Probabilities')
    plt.xticks([1, 2, 3, 4, 5, 6])  # Ensure x-axis goes from 1 to 6
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()

    plt.savefig('test correlations.png')



if __name__ == '__main__':
    try:
        fire.Fire(main())
    except:
        pass