import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import argparse
import fire
import pandas as pd
from tqdm import tqdm
import numpy as np
import scipy
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
import matplotlib.pyplot as plt
import seaborn as sns

from utils import SYSTEM_PROMPTS, SYSTEM_PROMPT_IDX, RATING_PROMPTS, RATING_PROMPT_IDX, CONVERSATIONS_DIR, RATINGS_DIR

# import models
from AG.models.huggingface.hf_inference_model import HFInferenceModel
from AG.models.openai.azure import AsyncAzureChatLLM
from AG.models.openai.gpt4 import GPT4Agent
from AG.models.vllm_models.inference_model import VLLMInferenceModel


# logging
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="conf", config_name='config')
def main(args: DictConfig) -> None:
    logging.info("Loading model for plots and analysis...")
    random.seed(1)
    
    
    with open(CONVERSATIONS_DIR, "r") as f:
        conversations = json.load(f)
    with open(RATINGS_DIR, "r") as f:
        ratings = json.load(f)
    

    # PER PERSONA AVERAGE RATING - this list will give a single average rating for each of the len(personas) different personas
    persona_average_ratings = []
    persona_average_ratings_SEMS = []
    for persona_id in range(5):
        agg_conversation_ratings = []
        for prompt_id in range(20):
            agg_conversation_ratings.extend(ratings[f"prompt-{prompt_id} persona-{persona_id}"])
        persona_average_ratings.append(1.0 * sum(agg_conversation_ratings) / len(agg_conversation_ratings))
        persona_average_ratings_SEMS.append(scipy.stats.sem(agg_conversation_ratings))
            
    
    # PER PERSONA AVERAGE DIFF BETWEEN MIN AND MAX RATING 
    persona_average_maxmindiff = []
    persona_average_maxmindiff_SEMS = []
    for persona_id in range(5):
        per_prompt_maxmindiffs = []
        for prompt_id in range(20):
            per_prompt_maxmindiffs.append(max(ratings[f"prompt-{prompt_id} persona-{persona_id}"]) - min(ratings[f"prompt-{prompt_id} persona-{persona_id}"]))
        persona_average_maxmindiff.append(1.0 * sum(per_prompt_maxmindiffs) / len(per_prompt_maxmindiffs))
        persona_average_maxmindiff_SEMS.append(scipy.stats.sem(per_prompt_maxmindiffs))
        
        
    # COUNT TURNS
    persona_avg_turns = []
    persona_avg_turns_SEMS = []
    for persona_id in range(5):
        per_prompt_turns = []
        for prompt_id in range(20):
            for conversation in conversations[f"prompt-{prompt_id} persona-{persona_id}"]:
                per_prompt_turns.append(conversation.count('[/INST]') - 1)
        persona_avg_turns.append(1.0 * sum(per_prompt_turns) / len(per_prompt_turns))
        persona_avg_turns_SEMS.append(scipy.stats.sem(per_prompt_turns))
    
    # Number of groups and bar width
    n_groups = 5
    bar_width = 0.3
    opacity = 0.8


    sns.set_theme(style="darkgrid")
    colors = sns.palettes.color_palette("colorblind", 10)
    #plt.figure(figsize=(12, 6))
    # Creating the figure and axes
    fig, ax = plt.subplots(figsize=(12, 7))
    

    # Index for the groups
    index = np.arange(n_groups)

    # Plotting the bars with error bars
    bars_averages = ax.bar(index, persona_average_ratings, bar_width, yerr=persona_average_ratings_SEMS, label='Average Rating', alpha=opacity, color=colors[0],)
    bars_diffs = ax.bar(index + bar_width, persona_average_maxmindiff, bar_width, yerr=persona_average_maxmindiff_SEMS, label='Average Max-Min rating Difference', alpha=opacity, color=colors[1],)
    bars_num_turns = ax.bar(index + 2 * bar_width, persona_avg_turns, bar_width, yerr=persona_avg_turns_SEMS, label='Average Number of Turns', alpha=opacity, color=colors[2],)

    # Adding labels and title
    ax.set_xlabel('Personas')
    ax.set_ylabel('Values')
    ax.set_ylim(0, 10)
    ax.set_title('Values by Persona and Metric')
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels([f'Persona {i}' for i in range(n_groups)])
    ax.legend()

    # Displaying the plot
    plt.tight_layout()
    plt.savefig('figures/test.png')
    
    
    
    
    

if __name__ == '__main__':
    try:
        fire.Fire(main())
    except:
        pass