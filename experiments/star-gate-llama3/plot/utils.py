import numpy as np
from paths import *
import json


PREFIX = f'{LOGPROBS_PATH}/{LLAMA_VERSION}/'


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

