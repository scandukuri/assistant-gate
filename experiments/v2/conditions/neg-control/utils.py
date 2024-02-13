from typing import List

GOLD_DIR = ''
NAMES_DIR = '/sailhome/andukuri/research_projects/assistant-gate/experiments/v2/persona-generation/new-personas/NAMES_EDITED_SYS-0_PROMPT-1_temp-1.1_topP-0.9_n-1_shotgroups-20.json'
PROMPTS_DIR = '/sailhome/andukuri/research_projects/assistant-gate/experiments/v2/instruct-questions/public/prompts.json'
PERSONAS_DIR = '/sailhome/andukuri/research_projects/assistant-gate/experiments/v2/persona-generation/new-personas/EDITED_SYS-0_PROMPT-1_temp-1.1_topP-0.9_n-1_shotgroups-20.json'


def flatten_list(
    lists: List[List]
    ) -> List:
    return [item for sublist in lists for item in sublist]