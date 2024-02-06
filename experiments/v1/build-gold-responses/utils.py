from typing import List

PERSONAS_DIR = '/sailhome/andukuri/research_projects/assistant-gate/experiments/v1/persona-generation/new-personas/SYS-0_PROMPT-1_temp-0.7_topP-0.9_n-2_shotgroups-5.json'
PROMPTS_DIR = '/sailhome/andukuri/research_projects/assistant-gate/experiments/v1/instruct-questions/first-10-prompts.json'



GENERATION_PROMPTS = [
    # 0
    """You are answering questions for the following user:

{}

Answer the question below, tailoring your answer to the user and their characteristics. Answer directly to the user (i.e., 'you', 'your' pronouns). In addition, incorporate aspects of their background when it is useful, but do not try to bring in aspects of the user's personality when they are irrelevant. Make sure to keep your answer concise and organized, but thorough. Keep your response to 10 sentences or less, using numbered lists where appropriate.

{}""",

    # 1
    """You are answering questions for the following user:

{}

Answer the question below, tailoring your answer to the user and their characteristics. Answer directly to the user (i.e., 'you', 'your' pronouns). In addition, incorporate aspects of their background when it is useful, but do not try to bring in aspects of the user's personality when they are irrelevant. Make sure to keep your answer concise and organized, but thorough. Keep your response to 15 sentences or less, using numbered lists where appropriate.

{}""",

    # 2
     """You are answering questions for the following user:

{}

Answer the question below, tailoring your answer to the user and their characteristics. Answer directly to the user (i.e., 'you', 'your' pronouns). In addition, incorporate aspects of their background when it is useful, but do not try to bring in aspects of the user's personality when they are irrelevant. Make sure to keep your answer concise and organized, but thorough. Keep your response to 3 sentences or less, using lists where appropriate.

{}""",

    # 3
     """You are answering questions for the following user:

{}

Answer the question below, tailoring your answer to the user and their characteristics. Answer directly to the user (i.e., 'you', 'your' pronouns). In addition, incorporate aspects of their background when it is useful, but do not try to bring in aspects of the user's personality when they are irrelevant. Make sure to keep your answer concise and organized, but thorough. Keep your response to 5 sentences or less, using numbered lists where appropriate.

{}"""
]
GENERATION_PROMPT_IDX = 0



SYS_PROMPTS = [
    """You are a helpful AI assistant, particularly skilled at providing personalized, satisfying answers to users given information about their background. You are able to construct responses that are tailored to their profession, hobbies, interests, relationships, locations, likes/dislikes and more, while maintaining a natural tone."""
]
SYS_PROMPT_IDX = 0



def flatten_list(
    lists: List[List]
    ) -> List:
    return [item for sublist in lists for item in sublist]