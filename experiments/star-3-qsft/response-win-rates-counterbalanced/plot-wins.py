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
    logging.info(f"Plotting win rates for {args.split.name}...")
    random.seed(1)
    
    t1_winrates, t2_winrates, t3_winrates, overall_winrates = list(), list(), list(), list()
    for i in range(args.N_ITER):
        overall_n, overall_d = 0, 0
        with open(f"{WINRATE_PATH}/{VERSION_3_QSFT}/baseline_m{i}/{args.split.name}_turn-1_win-rates.json", "r") as f:
            dct = json.load(f)
            print(dct)
            wins = Counter([rating[rating.find('Final Response:') + len('Final Response:'):].lower().strip() for key, rating in dct.items()])
            t1_winrates.append(wins['a']/(wins['a'] + wins['b']))
            overall_n += wins['a']
            overall_d += wins['a'] + wins['b']
        with open(f"{WINRATE_PATH}/{VERSION_3_QSFT}/baseline_m{i}/{args.split.name}_turn-2_win-rates.json", "r") as f:
            dct = json.load(f)
            print(dct)
            wins = Counter([rating[rating.find('Final Response:') + len('Final Response:'):].lower().strip() for key, rating in dct.items()])
            t2_winrates.append(wins['a']/(wins['a'] + wins['b']))
            overall_n += wins['a']
            overall_d += wins['a'] + wins['b']
        with open(f"{WINRATE_PATH}/{VERSION_3_QSFT}/baseline_m{i}/{args.split.name}_turn-3_win-rates.json", "r") as f:
            dct = json.load(f)
            print(dct)
            wins = Counter([rating[rating.find('Final Response:') + len('Final Response:'):].lower().strip() for key, rating in dct.items()])
            t3_winrates.append(wins['a']/(wins['a'] + wins['b']))
            overall_n += wins['a']
            overall_d += wins['a'] + wins['b']
        print(overall_d)
        overall_winrates.append(overall_n/overall_d)
    
    # Assuming the x-axis represents the iteration number
    iterations = list(range(len(t1_winrates)))



    # Plotting the win rates
    plt.figure(figsize=(10, 8))
    ax = plt.axes()
    ax.set_facecolor("whitesmoke")
    plt.grid(True, color='white', linestyle='-', linewidth=0.9)

    # Plot each set of win rates
    plt.plot(iterations[1:], t1_winrates[1:], marker='o', label='Turn 1 Win Rates', color='#00A86B')
    plt.plot(iterations[1:], t2_winrates[1:], marker='s', label='Turn 2 Win Rates', color='#65cbe9')
    plt.plot(iterations[1:], t3_winrates[1:], marker='^', label='Turn 3 Win Rates', color='#6c8dfa')
    plt.plot(iterations[1:], overall_winrates[1:], marker='*', label='Overall Win Rates', color='#ff6242')

    with open('star-3-qsft-overall-winrates.json', 'w') as f:
        json.dump(overall_winrates[1:], f)
    # Adding titles and labels
    plt.title('m_t Win Rates over baseline model')
    plt.xlabel('Iteration Number')
    plt.ylabel('Win Rate')
    plt.xticks(iterations)  # Set x-ticks to be iteration numbers
    plt.yticks([i/10 for i in range(5,10)])  # Assuming the win rate ranges from 0.1 to 0.9
    plt.xlim(0.5, 3.5)  # Set x-axis limits
    # Adding a legend
    plt.legend()

    # Show grid
    plt.grid(True)

    # Display the plot
    plt.savefig(f'{WINRATE_PATH}/{VERSION_3_QSFT}/{VERSION_3_QSFT}_counterbalanced-winrates-temp-{args.answer_model.run.completion_config.temperature}.png')
    
        
        
    
    
if __name__ == '__main__':
    try:
        fire.Fire(main())
    except:
        pass

