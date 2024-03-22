import numpy as np
from paths import *
import json


PREFIX = f'{LOGPROBS_PATH}/{VERSION_2_BSFT}/'
SPECIFIC_PROMPTS = [1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 18, 19, 20, 22, 23, 24, 27, 29, 30, 31, 32, 33, 34, 35, 38, 40, 42, 43, 45, 47, 48]
GENERAL_PROMPTS =[0, 3, 10, 16, 17, 21, 25, 26, 28, 36, 37, 39, 41, 44, 46, 49]


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

