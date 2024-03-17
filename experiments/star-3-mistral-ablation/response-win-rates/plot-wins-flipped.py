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
from collections import defaultdict, Counter
from datasets import load_dataset, Dataset
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

from paths import *
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
    logging.info(f"Loading model {args.qa_model.shortname} for response generation and win-rate computation for {args.split.name}...")
    random.seed(1)
    
    t1_winrates, t2_winrates, t3_winrates, overall_winrates = list(), list(), list(), list()
    for i in range(args.N_ITER):
        counts = list()
        overall_n, overall_d = 0, 0
        with open(f"{WINRATE_PATH}/{VERSION_3_MISTRAL_ABLATION}/m{i}_m0/{args.split.name}_turn-1_win-rates.json", "r") as f:
            dct = json.load(f)
            print(dct)
            wins = Counter([rating[rating.find('Final Response:') + len('Final Response:'):].lower().strip() for key, rating in dct.items()])
            t1_winrates.append(wins['a']/(wins['a'] + wins['b']))
            overall_n += wins['a']
            overall_d += wins['a'] + wins['b']
        with open(f"{WINRATE_PATH}/{VERSION_3_MISTRAL_ABLATION}/m{i}_m0/{args.split.name}_turn-2_win-rates.json", "r") as f:
            dct = json.load(f)
            print(dct)
            wins = Counter([rating[rating.find('Final Response:') + len('Final Response:'):].lower().strip() for key, rating in dct.items()])
            t2_winrates.append(wins['a']/(wins['a'] + wins['b']))
            overall_n += wins['a']
            overall_d += wins['a'] + wins['b']
        with open(f"{WINRATE_PATH}/{VERSION_3_MISTRAL_ABLATION}/m{i}_m0/{args.split.name}_turn-3_win-rates.json", "r") as f:
            dct = json.load(f)
            print(dct)
            wins = Counter([rating[rating.find('Final Response:') + len('Final Response:'):].lower().strip() for key, rating in dct.items()])
            t3_winrates.append(wins['a']/(wins['a'] + wins['b']))
            overall_n += wins['a']
            overall_d += wins['a'] + wins['b']
        overall_winrates.append(overall_n/overall_d)
    
    # Assuming the x-axis represents the iteration number
    iterations = list(range(len(t1_winrates)))

    # Plotting the win rates
    plt.figure(figsize=(10, 5))
    ax = plt.axes()
    ax.set_facecolor("whitesmoke")
    plt.grid(True, color='white', linestyle='-', linewidth=0.9)

    # Plot each set of win rates
    plt.plot(iterations, t1_winrates, marker='o', label='Turn 1 Win Rates')
    plt.plot(iterations, t2_winrates, marker='s', label='Turn 2 Win Rates')
    plt.plot(iterations, t3_winrates, marker='^', label='Turn 3 Win Rates')
    plt.plot(iterations, overall_winrates, marker='*', label='Overall Win Rates')

    # Adding titles and labels
    plt.title('m_t Win Rates over baseline model (m_t response first)')
    plt.xlabel('Iteration Number')
    plt.ylabel('Win Rate')
    plt.xticks(iterations)  # Set x-ticks to be iteration numbers
    plt.yticks([i/10 for i in range(1,10)])  # Assuming the win rate ranges from 0.1 to 0.9

    # Adding a legend
    plt.legend()

    # Show grid
    plt.grid(True)

    # Display the plot
    plt.savefig(f'{WINRATE_PATH}/{VERSION_3_MISTRAL_ABLATION}/winrates-temp-{args.qa_model.run.completion_config.temperature}-flipped.png')
    
        
        
    
    
if __name__ == '__main__':
    try:
        fire.Fire(main())
    except:
        pass