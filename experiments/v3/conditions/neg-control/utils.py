from typing import List

GOLD_DIR = '/sailhome/andukuri/research_projects/assistant-gate/experiments/v3/build-gold-responses/gold-responses/private/gen-0_sys-0_temp-0.json'
NAMES_DIR = '/sailhome/andukuri/research_projects/assistant-gate/experiments/v3/persona-generation/new-personas/private/NAMES_SYS-0_PROMPT-1_temp-1.1_topP-0.9_n-1_shotgroups-5.json'
PROMPTS_DIR = '/sailhome/andukuri/research_projects/assistant-gate/experiments/v3/instruct-questions/private/prompts.json'
PERSONAS_DIR = '/sailhome/andukuri/research_projects/assistant-gate/experiments/v3/persona-generation/new-personas/private/SYS-0_PROMPT-1_temp-1.1_topP-0.9_n-1_shotgroups-5.json'
CONVERSATIONS_DIR = '/sailhome/andukuri/research_projects/assistant-gate/experiments/v3/conditions/qa-experimental/simulated-conversations/private/m1_qa-12_humansys-1_human-4_maxturns-4.json'

# FILTERING PARAMETERS
k = 2
LOGPROBS_DIR = '/sailhome/andukuri/research_projects/assistant-gate/experiments/v2/conditions/neg-control/log-probs/model-mistral-7b-instruct-v02-vllm.json'


def flatten_list(
    lists: List[List]
    ) -> List:
    return [item for sublist in lists for item in sublist]