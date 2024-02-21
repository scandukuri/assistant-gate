from typing import List

PERSONAS_DIR = '/sailhome/andukuri/research_projects/assistant-gate/experiments/v2/persona-generation/new-personas/EDITED_SYS-0_PROMPT-1_temp-1.1_topP-0.9_n-1_shotgroups-20.json'
PROMPTS_DIR = '/sailhome/andukuri/research_projects/assistant-gate/experiments/v2/instruct-questions/public/prompts.json'
SPLIT = 'public'

GENERATION_PROMPTS = [
    # 0
    """You are answering questions for the following user:

{}

Answer the question below, tailoring your answer to the user and their characteristics. Answer directly to the user (i.e., 'you', 'your' pronouns). In addition, incorporate aspects of their background when it is useful, but do not try to bring in aspects of the user's personality when they are irrelevant. Make sure to keep your answer concise and organized, but thorough. Keep your response to 10 sentences or less, using numbered lists where appropriate.

{}""",
    # 1
    """You are answering questions for the following user:

{}

Answer the question below, tailoring your answer to the user and their characteristics. Answer directly to the user (i.e., 'you', 'your' pronouns). In addition, incorporate aspects of their background when it is useful, but do not try to bring in aspects of the user's personality when they are irrelevant. Make sure to keep your answer concise and organized, but thorough. Keep your response to ten sentences or less, and keep your response organized and clear.

{}"""
]
GENERATION_PROMPT_IDX = 1


SYS_PROMPTS = [
    """You are a helpful AI assistant, particularly skilled at providing personalized, satisfying answers to users given information about their background. You are able to construct responses that are tailored to their profession, hobbies, interests, relationships, locations, likes/dislikes and more, while maintaining a natural tone."""
]
SYS_PROMPT_IDX = 0


def flatten_list(
    lists: List[List]
    ) -> List:
    return [item for sublist in lists for item in sublist]


# Function to divide prompts into batches of size k
def batch_list(
    lst: List, 
    k: int
    ):
    for i in range(0, len(lst), k):
        yield lst[i:i + k]