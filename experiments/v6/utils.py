import numpy as np
k = 1


M0_POS_CONTROL_DIR = '/sailhome/andukuri/research_projects/assistant-gate/experiments/v6/log-probs/pos-control/m0_test.json'
M0_TOPK_POS_CONTROL_DIR = '/sailhome/andukuri/research_projects/assistant-gate/experiments/v6/log-probs/pos-control/m0_test_top-k-1.json'
M0_EXPERIMENTAL_DIR = '/sailhome/andukuri/research_projects/assistant-gate/experiments/v6/log-probs/qa-experimental/m0_test.json'
M0_TOPK_EXPERIMENTAL_DIR = '/sailhome/andukuri/research_projects/assistant-gate/experiments/v6/log-probs/qa-experimental/m0_test_top-k-1.json'
M0_NEG_CONTROL_DIR = '/sailhome/andukuri/research_projects/assistant-gate/experiments/v6/log-probs/neg-control/m0_test.json'


M0C_POS_CONTROL_DIR = '/sailhome/andukuri/research_projects/assistant-gate/experiments/v6/log-probs/pos-control/m0-confusion_test.json'
M0C_TOPK_POS_CONTROL_DIR = '/sailhome/andukuri/research_projects/assistant-gate/experiments/v6/log-probs/pos-control/m0-confusion_test_top-k-1.json'
M0C_EXPERIMENTAL_DIR = '/sailhome/andukuri/research_projects/assistant-gate/experiments/v6/log-probs/qa-experimental/m0-confusion_test.json'
M0C_TOPK_EXPERIMENTAL_DIR = '/sailhome/andukuri/research_projects/assistant-gate/experiments/v6/log-probs/qa-experimental/m0-confusion_test_top-k-1.json'
M0C_NEG_CONTROL_DIR = '/sailhome/andukuri/research_projects/assistant-gate/experiments/v6/log-probs/neg-control/m0-confusion_test.json'


M1_POS_CONTROL_DIR = '/sailhome/andukuri/research_projects/assistant-gate/experiments/v6/log-probs/pos-control/m1_test.json'
M1_TOPK_POS_CONTROL_DIR = '/sailhome/andukuri/research_projects/assistant-gate/experiments/v6/log-probs/pos-control/m1_test_top-k-1.json'
M1_EXPERIMENTAL_DIR = '/sailhome/andukuri/research_projects/assistant-gate/experiments/v6/log-probs/qa-experimental/m1_test.json'
M1_TOPK_EXPERIMENTAL_DIR = '/sailhome/andukuri/research_projects/assistant-gate/experiments/v6/log-probs/qa-experimental/m1_test_top-k-1.json'
M1_NEG_CONTROL_DIR = '/sailhome/andukuri/research_projects/assistant-gate/experiments/v6/log-probs/neg-control/m0_test.json'

M1C_POS_CONTROL_DIR = '/sailhome/andukuri/research_projects/assistant-gate/experiments/v6/log-probs/pos-control/m1-confusion_test.json'
M1C_TOPK_POS_CONTROL_DIR = '/sailhome/andukuri/research_projects/assistant-gate/experiments/v6/log-probs/pos-control/m1-confusion_test_top-k-1.json'
M1C_EXPERIMENTAL_DIR = '/sailhome/andukuri/research_projects/assistant-gate/experiments/v6/log-probs/qa-experimental/m1-confusion_test.json'
M1C_TOPK_EXPERIMENTAL_DIR = '/sailhome/andukuri/research_projects/assistant-gate/experiments/v6/log-probs/qa-experimental/m1-confusion_test_top-k-1.json'
M1C_NEG_CONTROL_DIR = '/sailhome/andukuri/research_projects/assistant-gate/experiments/v6/log-probs/neg-control/m1-confusion_test.json'



## FILEPATHS
SIMULATION_PATH = '/scr/andukuri/assistant-gate-hgx/simulated-conversations'
LOGPROBS_PATH = '/scr/andukuri/assistant-gate-hgx/log-probs'
VERSION = 'v6'


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