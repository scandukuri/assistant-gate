from typing import List

NAMES_DIR = '/sailhome/andukuri/research_projects/assistant-gate/experiments/v1/persona-generation/new-personas/SYS-0_PROMPT-1_temp-0.7_topP-0.9_n-2_shotgroups-5_NAMES.json'
PERSONAS_DIR = '/sailhome/andukuri/research_projects/assistant-gate/experiments/v1/persona-generation/new-personas/SYS-0_PROMPT-1_temp-0.7_topP-0.9_n-2_shotgroups-5.json'
PROMPTS_DIR = '/sailhome/andukuri/research_projects/assistant-gate/experiments/v1/instruct-questions/first-10-prompts.json'
GOLD_DIR = '/sailhome/andukuri/research_projects/assistant-gate/experiments/v1/build-gold-responses/gold-responses/model-gpt4_gen-0_sys-0_temp-0_n-1.json'


def flatten_list(
    lists: List[List]
    ) -> List:
    return [item for sublist in lists for item in sublist]