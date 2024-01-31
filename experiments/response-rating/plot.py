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
    

    invalid_count = 0
    # PER PERSONA AVERAGE RATING - this list will give a single average rating for each of the len(personas) different personas
    persona_average_ratings = []
    persona_average_ratings_SEMS = []
    for persona_id in range(5):
        agg_conversation_ratings = []
        for prompt_id in range(50):
            if len(ratings[f"prompt-{prompt_id} persona-{persona_id}"]) < 5 or -1 in ratings[f"prompt-{prompt_id} persona-{persona_id}"]:
                invalid_count += 1
                continue
            agg_conversation_ratings.extend(ratings[f"prompt-{prompt_id} persona-{persona_id}"])
        persona_average_ratings.append(1.0 * sum(agg_conversation_ratings) / len(agg_conversation_ratings))
        persona_average_ratings_SEMS.append(scipy.stats.sem(agg_conversation_ratings))
            
    
    # PER PERSONA AVERAGE DIFF BETWEEN MIN AND MAX RATING 
    persona_average_maxmindiff = []
    persona_average_maxmindiff_SEMS = []
    for persona_id in range(5):
        per_prompt_maxmindiffs = []
        for prompt_id in range(50):
            if len(ratings[f"prompt-{prompt_id} persona-{persona_id}"]) < 5 or -1 in ratings[f"prompt-{prompt_id} persona-{persona_id}"]:
                continue
            per_prompt_maxmindiffs.append(max(ratings[f"prompt-{prompt_id} persona-{persona_id}"]) - min(ratings[f"prompt-{prompt_id} persona-{persona_id}"]))
        persona_average_maxmindiff.append(1.0 * sum(per_prompt_maxmindiffs) / len(per_prompt_maxmindiffs))
        persona_average_maxmindiff_SEMS.append(scipy.stats.sem(per_prompt_maxmindiffs))
        
        
    # COUNT TURNS
    persona_avg_turns = []
    persona_avg_turns_SEMS = []
    for persona_id in range(5):
        per_prompt_turns = []
        for prompt_id in range(50):
            if len(ratings[f"prompt-{prompt_id} persona-{persona_id}"]) < 5 or -1 in ratings[f"prompt-{prompt_id} persona-{persona_id}"]:
                continue
            for conversation in conversations[f"prompt-{prompt_id} persona-{persona_id}"]:
                per_prompt_turns.append(conversation.count('[/INST]') - 1)
        persona_avg_turns.append(1.0 * sum(per_prompt_turns) / len(per_prompt_turns))
        persona_avg_turns_SEMS.append(scipy.stats.sem(per_prompt_turns))
    
    # Number of groups and bar width
    n_groups = 5
    bar_width = 0.3
    opacity = 0.8
    index = np.arange(n_groups)


    sns.set_theme(style="darkgrid")
    colors = sns.palettes.color_palette("colorblind", 10)




    ####################################
    # CONFIGURATION 1
    ####################################

    # print(persona_average_ratings)
    # # Creating the figure and axes
    # fig, ax = plt.subplots(figsize=(7, 7))
    # # Plotting the bars with error bars
    # bars_averages = ax.bar(index, persona_average_ratings, bar_width, yerr=persona_average_ratings_SEMS, label='Average Rating', alpha=opacity, color=colors[0],)
    # ax.set_xlabel('Personas')
    # ax.set_ylabel('Values')
    # ax.set_ylim(0, 10)
    # ax.set_title('Average Rating by Persona')
    # ax.set_xticks(index)
    # ax.set_xticklabels([f'Persona {i}' for i in range(n_groups)])
    # plt.tight_layout()
    # plt.savefig('figures/average-ratings.png')
    
    
    # print(persona_average_maxmindiff)
    # # Creating the figure and axes
    # fig, ax = plt.subplots(figsize=(7, 7))
    # # Plotting the bars with error bars
    # bars_diffs = ax.bar(index, persona_average_maxmindiff, bar_width, yerr=persona_average_maxmindiff_SEMS, label='Average Difference between Min-Max Ratings', alpha=opacity, color=colors[1],)
    # ax.set_xlabel('Personas')
    # ax.set_ylabel('Values')
    # ax.set_ylim(0, 9)
    # ax.set_title('Average Difference between Min-Max Ratings by Persona')
    # ax.set_xticks(index)
    # ax.set_xticklabels([f'Persona {i}' for i in range(n_groups)])
    # plt.tight_layout()
    # plt.savefig('figures/average-maxmin-diffs.png')
    
    
    # print(persona_avg_turns)
    # # Creating the figure and axes
    # fig, ax = plt.subplots(figsize=(7, 7))
    # # Plotting the bars with error bars
    # bars_turns = ax.bar(index, persona_avg_turns, bar_width, yerr=persona_avg_turns_SEMS, label='Average # of Turns in Conversation', alpha=opacity, color=colors[2],)
    # ax.set_xlabel('Personas')
    # ax.set_ylabel('Values')
    # ax.set_ylim(0, 10)
    # ax.set_title('Average # of Turns in Conversation by Persona')
    # ax.set_xticks(index)
    # ax.set_xticklabels([f'Persona {i}' for i in range(n_groups)])
    # plt.tight_layout()
    # plt.savefig('figures/average-turns.png')
    








    ####################################
    # CONFIGURATION 2
    ####################################

    # # Index for the groups
    # index = np.arange(n_groups)

    # fig, ax = plt.subplots(figsize=(12, 7))
    # # Plotting the bars with error bars
    # bars_averages = ax.bar(index, persona_average_ratings, bar_width, yerr=persona_average_ratings_SEMS, label='Average Rating', alpha=opacity, color=colors[0],)
    # bars_diffs = ax.bar(index + bar_width, persona_average_maxmindiff, bar_width, yerr=persona_average_maxmindiff_SEMS, label='Average Max-Min Rating Difference', alpha=opacity, color=colors[1],)
    # bars_num_turns = ax.bar(index + 2 * bar_width, persona_avg_turns, bar_width, yerr=persona_avg_turns_SEMS, label='Average Number of Turns', alpha=opacity, color=colors[2],)

    # # Adding labels and title
    # ax.set_xlabel('Personas')
    # ax.set_ylabel('Values')
    # ax.set_ylim(0, 10)
    # ax.set_title('Values by Persona and Metric')
    # ax.set_xticks(index + bar_width)
    # ax.set_xticklabels([f'Persona {i}' for i in range(n_groups)])
    # ax.legend(loc='center right', bbox_to_anchor=(1.25, 0.5))
    
    # text_offset = 0.15
    # # Add values on top of the bars
    # for i, bar in enumerate(bars_averages):
    #     ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + text_offset, f'{persona_average_ratings[i]:.2f}', ha='center', va='bottom')
    # for i, bar in enumerate(bars_diffs):
    #     ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + text_offset, f'{persona_average_maxmindiff[i]:.2f}', ha='center', va='bottom')
    # for i, bar in enumerate(bars_num_turns):
    #     ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height()+ text_offset, f'{persona_avg_turns[i]:.2f}', ha='center', va='bottom')

    # plt.tight_layout()
    # plt.savefig('figures/test.png')
    
    
    
    
    
    
    
    
    
    
    ####################################
    # CONFIGURATION 3
    ####################################
     # Index for the groups
    index = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(12, 7))
    # Plotting the bars with error bars
    bars_averages = ax.bar(index, persona_average_ratings, bar_width, yerr=persona_average_ratings_SEMS, label='Average Rating', alpha=opacity, color=colors[0],)
    bars_diffs = ax.bar(index + bar_width, persona_average_maxmindiff, bar_width, yerr=persona_average_maxmindiff_SEMS, label='Average Max-Min Rating Difference', alpha=opacity, color=colors[1],)
    
    # Adding labels and title
    ax.set_xlabel('Personas')
    ax.set_ylabel('Values')
    ax.set_ylim(0, 10)
    ax.set_title('Average Ratings and Max-Min Rating Difference by Persona and Metric')
    ax.set_xticks(index + 0.5 * bar_width)
    ax.set_xticklabels([f'Persona {i}' for i in range(n_groups)])
    ax.legend(loc='center right', bbox_to_anchor=(1.25, 0.5))
    
    text_offset = 0.15
    for i, bar in enumerate(bars_averages):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + text_offset, f'{persona_average_ratings[i]:.2f}', ha='center', va='bottom')
    for i, bar in enumerate(bars_diffs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + text_offset, f'{persona_average_maxmindiff[i]:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('figures/averages-diffs.png')
    
    
    
    fig, ax = plt.subplots(figsize=(12, 7))
    bars_num_turns = ax.bar(index, persona_avg_turns, bar_width, yerr=persona_avg_turns_SEMS, label='Average # of QA Model Turns', alpha=opacity, color=colors[2],)

    # Adding labels and title
    ax.set_xlabel('Personas')
    ax.set_ylabel('Values')
    ax.set_ylim(0, 10)
    ax.set_title('Average QA Model Turns by Persona and Metric')
    ax.set_xticks(index)
    ax.set_xticklabels([f'Persona {i}' for i in range(n_groups)])
    ax.legend(loc='center right', bbox_to_anchor=(1.25, 0.5))
    
    text_offset = 0.15
    for i, bar in enumerate(bars_num_turns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height()+ text_offset, f'{persona_avg_turns[i]:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('figures/turns.png')
    
    print('Invalid simulation count: ', invalid_count)
    
    

if __name__ == '__main__':
    try:
        fire.Fire(main())
    except:
        pass