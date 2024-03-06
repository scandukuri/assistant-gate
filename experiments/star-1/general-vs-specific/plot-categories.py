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
    print(f"args N_ITER {args.N_ITER}")
    

    EXPERIMENTAL = [f'{PREFIX}/qa-experimental/m{i}/m0_{args.split.name}.json' for i in range(args.N_ITER)]
    TOPK_EXPERIMENTAL = [f'{PREFIX}/qa-experimental/m{i}/m0_{args.split.name}_top-k-{args.k}.json' for i in range(args.N_ITER)]
    
    with open(f'{PROMPT_PATH}/{VERSION}/{args.split.name}.json', "r") as f:
        prompts = json.load(f)
        prompts = [s.strip() for s in prompts]
    # Load personas
    with open(f'{PERSONAS_PATH}/{VERSION}/{args.split.name}.json', "r") as f:
        personas = json.load(f)
        
    iterations = list(range(args.N_ITER))

    # Load the log-probabilities from the files
    experimental = [load_file(pth) for pth in EXPERIMENTAL]
    topk_experimental = [load_file(pth) for pth in TOPK_EXPERIMENTAL]

    general_keys, specific_keys = flatten_list([[f'prompt-{prompt_idx} persona-{j}'for j in range(len(personas))] for prompt_idx in GENERAL_PROMPTS]), flatten_list([[f'prompt-{prompt_idx} persona-{j}'for j in range(len(personas))] for prompt_idx in SPECIFIC_PROMPTS])

    experimental_general, experimental_specific = [dict([(key, val) for key, val in dct.items() if key in general_keys]) for dct in experimental], [dict([(key, val) for key, val in dct.items() if key in specific_keys]) for dct in experimental]
    topk_experimental_general, topk_experimental_specific = [dict([(key, val) for key, val in dct.items() if key in general_keys]) for dct in topk_experimental], [dict([(key, val) for key, val in dct.items() if key in specific_keys]) for dct in topk_experimental]
    # Now calculate means and SEMs instead of just means
    
    experimental_general_means, experimental_general_sems = zip(*[calculate_mean_and_sem([lst for key, lst in logprobs.items()]) for logprobs in experimental_general])
    experimental_specific_means, experimental_specific_sems = zip(*[calculate_mean_and_sem([lst for key, lst in logprobs.items()]) for logprobs in experimental_specific])
    experimental_topk_general_means, experimental_topk_general_sems = zip(*[calculate_mean_and_sem([lst for key, lst in logprobs.items()]) for logprobs in topk_experimental_general])
    experimental_topk_specific_means, experimental_topk_specific_sems = zip(*[calculate_mean_and_sem([lst for key, lst in logprobs.items()]) for logprobs in topk_experimental_specific])
    
    

    # Adjusting the provided code to use plot() instead of scatter(), and making the background grid light gray
    
    # Adjust marker sizes for better visibility with plot()
    

    marker_size = 6

    # Plotting
    plt.figure(figsize=(5, 6))
    ax = plt.axes()
    ax.set_facecolor("gainsboro")

    plt.xticks(range(3))  # This will show ticks from 0 to 5
    plt.xlim(-0.5, 2.5)  # Set the limit so the x-axis will start a bit before 0 and end a bit after 5

    # Error bars
    ax.errorbar(iterations, experimental_general_means, yerr=experimental_general_sems, fmt='o', color='#e86464', label='General', markersize=marker_size, linestyle='--')
    ax.errorbar(iterations, experimental_specific_means, yerr=experimental_specific_sems, fmt='o', color='#4285f4', label='Specific', markersize=marker_size, linestyle='--')
    ax.errorbar(iterations, experimental_topk_general_means, yerr=experimental_topk_general_sems, fmt='o', color='#e86464', label=f'General Top-{args.k}', markersize=marker_size, alpha=0.4, linestyle=':')
    ax.errorbar(iterations, experimental_topk_specific_means, yerr=experimental_topk_specific_sems, fmt='o', color='#4285f4', label=f'Specific Top-{args.k}', markersize=marker_size, alpha=0.4, linestyle=':')

    # Title and labels
    ax.set_title('Log-Probability of Desired Results by Category')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Log-Probability')

    # Legend
    ax.legend()

    # Show plot
    plt.grid(True, color='white', linestyle='-', linewidth=0.9)
    plt.tight_layout()
    # Show the plot
    if not os.path.exists(f'{FIGURES_PATH}/{VERSION}'):
        os.makedirs(f'{FIGURES_PATH}/{VERSION}')
    plt.savefig(f'{FIGURES_PATH}/{VERSION}/m0_log_probs_general-vs-specific-{args.split.name}_top-{args.k}.png')



if __name__ == '__main__':
    try:
        fire.Fire(main())
    except:
        pass