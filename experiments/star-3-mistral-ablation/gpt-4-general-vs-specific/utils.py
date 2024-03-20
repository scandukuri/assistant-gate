import numpy as np
from paths import *
import json


RATING_SYS_PROMPTS = [
    # 0
    """You are an expert in evaluating user queries, particularly skilled in quantitatively analyzing whether a query is more suited to a general response regardless of the user, or more suited to a personalized and specific response dependent on the user's background and preferences."""
]

RATING_PROMPTS = [
    # 0
    """For the following user query to an assistant, rate how well-suited the user query is to a general response regardless of the user or to a personalized and specific response dependent on the user's background and preferences.
    
User Query: {}

FIRST, provide a thorough discussion of the user query and explain whether you feel the user query is more suited to a general response regardless of the user or to a personalized and specific response dependent on the user's background and preferences. (no more than 100 words). 

SECOND, on a new line, state only an integer between 1 and 6 inclusive to indicate your rating for the user query, where 1 = very user-specific, 2 = user-specific, 3 = slightly user-specific, 4 = slightly general, 5 = general, 6 = very general.

Comparison: <thorough discussion of the user query and explanation>

Final Response: <integer between 1 and 6, inclusive">"""
]

PREFIX = f'{LOGPROBS_PATH}/{VERSION_3_MISTRAL_ABLATION}/'

def flatten_list(lists):
    return [item for sublist in lists for item in sublist]

def dict_to_list(json_lists):
    return [lst for key, lst in json_lists.items()]


# Function to calculate mean and SEM from a list of values
def calculate_mean_and_sem(values):
    values = flatten_list(values)
    mean = np.mean(values)
    sem = np.std(values, ddof=1) / np.sqrt(len(values))  # Use ddof=1 for sample standard deviation
    return mean, sem


error_color = 'black'
z_order = 1000
marker_size = 11  # Adjust this value as needed for larger markers with plot()


def load_file(file):
    with open(file, 'r') as f:
        return json.load(f)


