from typing import List


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

{}""",

# 2
    """You are answering questions for the following user:

{}

Answer the question below, tailoring your answer to the user and their characteristics. Answer directly to the user (i.e., 'you', 'your' pronouns). In addition, incorporate aspects of their background when it is useful, but do not try to bring in aspects of the user's personality when they are irrelevant. Make sure to keep your answer concise and organized, but thorough. Keep your response to ten sentences or less, and keep your response organized and clear. Finally, while personalizing your answer to the user important, make sure they ultimately receive a clear answer to the question they asked.

{}"""
]



SYS_PROMPTS = [
    """You are a helpful AI assistant, particularly skilled at providing personalized, satisfying answers to users given information about their background. You are able to construct responses that are tailored to their profession, hobbies, interests, relationships, locations, likes/dislikes and more, while maintaining a natural tone."""
]



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