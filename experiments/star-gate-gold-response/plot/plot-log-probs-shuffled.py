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
    
    POS_CONTROL_1 = [f'{PREFIX}/pos-control-1/m{i}/{args.split.name}.json' for i in range(args.N_ITER)]
    TOPK_POS_CONTROL_1 = [f'{PREFIX}/pos-control-1/m{i}/{args.split.name}_top-k-{args.k}.json' for i in range(args.N_ITER)]
    POS_CONTROL_2 = [f'{PREFIX}/pos-control-2/m{i}/{args.split.name}.json' for i in range(args.N_ITER)]
    TOPK_POS_CONTROL_2 = [f'{PREFIX}/pos-control-2/m{i}/{args.split.name}_top-k-{args.k}.json' for i in range(args.N_ITER)]
    EXPERIMENTAL = [f'{PREFIX}/qa-experimental/m{i}/{args.split.name}-shuffled.json' for i in range(args.N_ITER)]
    TOPK_EXPERIMENTAL = [f'{PREFIX}/qa-experimental/m{i}/{args.split.name}-shuffled_top-k-{args.k}.json' for i in range(args.N_ITER)]
    NEG_CONTROL = [f'{PREFIX}/neg-control/m{i}/{args.split.name}.json' for i in range(args.N_ITER)]
    iterations = list(range(args.N_ITER))

    pos_control_1 = [load_file(pth) for pth in POS_CONTROL_1]
    pos_control_2 = [load_file(pth) for pth in POS_CONTROL_2]
    neg_control = [load_file(pth) for pth in NEG_CONTROL]
    topk_pos_control_1 = [load_file(pth) for pth in TOPK_POS_CONTROL_1]
    topk_pos_control_2 = [load_file(pth) for pth in TOPK_POS_CONTROL_2]
    experimental = [load_file(pth) for pth in EXPERIMENTAL]
    topk_experimental = [load_file(pth) for pth in TOPK_EXPERIMENTAL]


#     # Now calculate means and SEMs instead of just means
    pos_control_1_means, pos_control_1_sems = zip(*[calculate_mean_and_sem([lst for key, lst in logprobs.items()]) for logprobs in pos_control_1])
    pos_control_2_means, pos_control_2_sems = zip(*[calculate_mean_and_sem([lst for key, lst in logprobs.items()]) for logprobs in pos_control_2])
    pos_control_1_topk_means, pos_control_1_topk_sems = zip(*[calculate_mean_and_sem([lst for key, lst in logprobs.items()]) for logprobs in topk_pos_control_1])
    pos_control_2_topk_means, pos_control_2_topk_sems = zip(*[calculate_mean_and_sem([lst for key, lst in logprobs.items()]) for logprobs in topk_pos_control_2])
    experimental_means, experimental_sems = zip(*[calculate_mean_and_sem([lst for key, lst in logprobs.items()]) for logprobs in experimental])
    experimental_topk_means, experimental_topk_sems = zip(*[calculate_mean_and_sem([lst for key, lst in logprobs.items()]) for logprobs in topk_experimental])
    neg_control_means, neg_control_sems = zip(*[calculate_mean_and_sem([lst for key, lst in logprobs.items()]) for logprobs in neg_control])
    
    

    # Adjusting the provided code to use plot() instead of scatter(), and making the background grid light gray
    
    # Adjust marker sizes for better visibility with plot()
    

    # Define custom colors to match the provided image as closely as possible
    color_map = {
        'pos-control-1': '#FFA500',  # yellow color for positive control 1,
        'pos-control-2' : '#1fb37eff', # green color for positive control 2
        'neg-control': '#1F77B4',  # Muted blue color for negative control
        'qa-experimental': '#FF7F0E' # Safety orange for QA experimental
    }


    # Log-probability data points for each category at iteration 0
    log_probabilities = {
        'pos-control-1': pos_control_1_means,
        'pos-control-2': pos_control_2_means,
        'neg-control': neg_control_means,
        'qa-experimental': experimental_means,
        f'pos-control-1-top-{args.k}' : pos_control_1_topk_means,
        f'pos-control-2-top-{args.k}' : pos_control_2_topk_means,
        f'qa-experimental-top-{args.k}' : experimental_topk_means
    }

    # Set up the plot with larger figure size for better visibility of larger markers
    plt.figure(figsize=(10, 8))
    ax = plt.axes()
    ax.set_facecolor("whitesmoke")


#     # Plot each category with a larger marker using plot()
    plt.plot(iterations, log_probabilities['pos-control-1'], label='pos-control-1', 
            marker='^', markersize=marker_size, color=color_map['pos-control-1'], linestyle='--', zorder=z_order/100)
    plt.plot(iterations, log_probabilities[f'pos-control-1-top-{args.k}'], label=f'pos-control-1-top-{args.k}', 
            marker='^', markersize=marker_size, color=color_map['pos-control-1'], linestyle=':', alpha=0.4, zorder=z_order/100)
    plt.plot(iterations, log_probabilities['pos-control-2'], label='pos-control-2', 
            marker='^', markersize=marker_size, color=color_map['pos-control-2'], linestyle='', zorder=z_order/100)
    plt.plot(iterations, log_probabilities[f'pos-control-2-top-{args.k}'], label=f'pos-control-2-top-{args.k}', 
            marker='^', markersize=marker_size, color=color_map['pos-control-2'], linestyle=':', alpha=0.4, zorder=z_order/100)
    plt.plot(iterations, log_probabilities['neg-control'], label='neg-control', 
            marker='s', markersize=marker_size, color=color_map['neg-control'], linestyle='--', zorder=z_order/100)
    plt.plot(iterations, log_probabilities['qa-experimental'], label='shuffled-qa-experimental', 
            marker='o', markersize=marker_size, color=color_map['qa-experimental'], linestyle='--', zorder=z_order/100)
    plt.plot(iterations, log_probabilities[f'qa-experimental-top-{args.k}'], label=f'shuffled-qa-experimental-top-{args.k}', 
            marker='o', markersize=marker_size, color=color_map['qa-experimental'], linestyle=':', alpha=0.4, zorder=z_order/100)
    


    # # Plot each category with error bars
    plt.errorbar(iterations, pos_control_1_means, yerr=pos_control_1_sems, marker='^', markersize=marker_size, color=color_map['pos-control-1'], linestyle='', capsize=0, ecolor=error_color, zorder=z_order, fmt='none', label='_nolegend_', elinewidth=1)
    plt.errorbar(iterations, pos_control_2_means, yerr=pos_control_2_sems, marker='^', markersize=marker_size, color=color_map['pos-control-2'], linestyle='', capsize=0, ecolor=error_color, zorder=z_order, fmt='none', label='_nolegend_', elinewidth=1)
    plt.errorbar(iterations, neg_control_means, yerr=neg_control_sems, marker='s', markersize=marker_size, color=color_map['neg-control'], linestyle='', capsize=0,  ecolor=error_color, zorder=z_order, fmt='none', label='_nolegend_', elinewidth=1)
    plt.errorbar(iterations, experimental_means, yerr=experimental_sems, marker='o', markersize=marker_size, color=color_map['qa-experimental'], linestyle='',  ecolor=error_color, zorder=z_order, fmt='none', label='_nolegend_', elinewidth=1)
    
    # If the top-k categories also need error bars, adjust similarly
    # For example:
    plt.errorbar(iterations, pos_control_1_topk_means, yerr=pos_control_1_topk_sems, marker='^', markersize=marker_size, color=color_map['pos-control-1'], linestyle='', capsize=0,  ecolor=error_color, zorder=z_order, fmt='none', label='_nolegend_', elinewidth=1)
    plt.errorbar(iterations, pos_control_2_topk_means, yerr=pos_control_2_topk_sems, marker='^', markersize=marker_size, color=color_map['pos-control-2'], linestyle='', capsize=0,  ecolor=error_color, zorder=z_order, fmt='none', label='_nolegend_', elinewidth=1)    
    plt.errorbar(iterations, experimental_topk_means, yerr=experimental_topk_sems, marker='o', markersize=marker_size, color=color_map['qa-experimental'], linestyle='', capsize=0,  ecolor=error_color, zorder=z_order, fmt='none', label='_nolegend_', elinewidth=1)


    # with open('star-1-esft-pos_control_1_means.json', 'w') as f:
    #     json.dump(pos_control_1_means, f)
    # with open('star-1-esft-pos_control_1_sems.json', 'w') as f:
    #     json.dump(pos_control_1_sems, f)
    # with open('star-1-esft-pos_control_1_topk_means.json', 'w') as f:
    #     json.dump(pos_control_1_topk_means, f)
    # with open('star-1-esft-pos_control_1_topk_sems.json', 'w') as f:
    #     json.dump(pos_control_1_topk_sems, f)
    # with open('star-1-esft-pos_control_2_means.json', 'w') as f:
    #     json.dump(pos_control_2_means, f)
    # with open('star-1-esft-pos_control_2_sems.json', 'w') as f:
    #     json.dump(pos_control_2_sems, f)
    # with open('star-1-esft-pos_control_2_topk_means.json', 'w') as f:
    #     json.dump(pos_control_2_topk_means, f)
    # with open('star-1-esft-pos_control_2_topk_sems.json', 'w') as f:
    #     json.dump(pos_control_2_topk_sems, f)
    with open('star-1-esft-qa-experimental-shuffled_means.json', 'w') as f:
        json.dump(experimental_means, f)
    with open('star-1-esft-qa-experimental-shuffled_sems.json', 'w') as f:
        json.dump(experimental_sems, f)
    with open('star-1-esft-qa-experimental-shuffled_topk_means.json', 'w') as f:
        json.dump(experimental_topk_means, f)
    with open('star-1-esft-qa-experimental-shuffled_topk_sems.json', 'w') as f:
        json.dump(experimental_topk_sems, f)
    # with open('star-1-esft-neg-control_means.json', 'w') as f:
    #     json.dump(neg_control_means, f)
    # with open('star-1-esft-neg-control_sems.json', 'w') as f:
    #     json.dump(neg_control_sems, f)
              
    
    # Set the x-axis to show the range up to iteration 5
    plt.xticks(range(4))  # This will show ticks from 0 to 5
    plt.xlim(-0.5, 5.5)  # Set the limit so the x-axis will start a bit before 0 and end a bit after 5

    # Set x-axis range
    ax.set_ylim([-375, -250])

    # Set the labels for x and y axes
    plt.xlabel('Iterations')
    plt.ylabel('Log-Probability')

    # Add the legend to the plot
    plt.legend(borderpad=1, fontsize='large')
    

# Show grid with light gray color
    plt.grid(True, color='white', linestyle='-', linewidth=0.9)
    plt.title(f'{VERSION_1_ESFT}\nAverage SHUFFLED Log-Probability of Desired Responses in {args.split.name} Split\n[n = {5000} simulations]')

# Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()

# Show the plot
    if not os.path.exists(f'{FIGURES_PATH}/{VERSION_1_ESFT}'):
        os.makedirs(f'{FIGURES_PATH}/{VERSION_1_ESFT}')
    # the file title is on purpose, the redundancy of VERSION_1_ESFT helps identify the experiment when we copy over outside of the regular working node (i.e. to send in Slack etc.)
    plt.savefig(f'{FIGURES_PATH}/{VERSION_1_ESFT}/{VERSION_1_ESFT}-log_probs_{args.split.name}-shuffled_top-{args.k}.png')



if __name__ == '__main__':
    try:
        fire.Fire(main())
    except:
        pass