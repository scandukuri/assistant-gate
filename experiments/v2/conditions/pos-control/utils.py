from typing import List

GOLD_DIR = '/sailhome/andukuri/research_projects/assistant-gate/experiments/v2/build-gold-responses/gold-responses/public/model-mistral-7b-instruct-v02-vllm_gen-1_sys-0_temp-0.json'
NAMES_DIR = '/sailhome/andukuri/research_projects/assistant-gate/experiments/v2/persona-generation/new-personas/NAMES_EDITED_SYS-0_PROMPT-1_temp-1.1_topP-0.9_n-1_shotgroups-20.json'
PROMPTS_DIR = '/sailhome/andukuri/research_projects/assistant-gate/experiments/v2/instruct-questions/public/prompts.json'
PERSONAS_DIR = '/sailhome/andukuri/research_projects/assistant-gate/experiments/v2/persona-generation/new-personas/EDITED_SYS-0_PROMPT-1_temp-1.1_topP-0.9_n-1_shotgroups-20.json'
CONVERSATIONS_DIR = '/sailhome/andukuri/research_projects/assistant-gate/experiments/v2/conditions/qa-experimental/simulated-conversations/public/qa-model-mistral-7b-instruct-v02-vllm_human-model-mistral-7b-instruct-v02-vllm_qa-12_humansys-1_human-4_maxturns-4.json'

# FILTERING PARAMS
LOGPROBS_DIR = '/sailhome/andukuri/research_projects/assistant-gate/experiments/v2/conditions/pos-control/log-probs/logprobs-poscontrol-qa-model-mistral-7b-instruct-v02-vllm_human-model-mistral-7b-instruct-v02-vllm_qa-12_humansys-1_human-4_maxturns-4.json'
INDICES_DIR = '/sailhome/andukuri/research_projects/assistant-gate/experiments/v2/conditions/qa-experimental/log-probs/logprobs-qa-model-mistral-7b-instruct-v02-vllm_human-model-mistral-7b-instruct-v02-vllm_qa-12_humansys-1_human-4_maxturns-4_top-k-3_indices.json'
k = 3



def flatten_list(
    lists: List[List]
    ) -> List:
    return [item for sublist in lists for item in sublist]