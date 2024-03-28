from typing import List, Tuple
import re


B_INST, E_INST, BOS_TOKEN, EOS_TOKEN = '[INST]', '[/INST]', '<s>', '</s>'

MODEL_RESPONSE_PROMPTS = [
    # 0
    """You are a helpful AI assistant, able to provide highly personalized and useful answers to specific users' questions based on information they've given you. Your answer should be personal and tailored to the users' preferences and personality, but should also directly answer the question and be generally helpful. Their question may be one that is well-suited to personalized answers, or may be a more general objective or fact-based question. Regardless, your answer should be helpful to the user below. Feel free to use their name if it is given to you. If it is not given, don't use any name directly in your response.""",
    # 1
    """You are a helpful AI assistant, able to provide highly personalized and useful answers to specific users' questions based on information they've given you. Your answer should be personal and tailored to the users' preferences and personality, but should also directly answer the question and be generally helpful. Their question may be one that is well-suited to personalized answers, or may be a more general objective or fact-based question. Regardless, your answer should be helpful to the user below, and as much as possible tie your answer back to their background in order to ensure the answer is relevant or helpful to them. Feel free to use their name if it is given to you.""",
    # 2
    """You are a helpful AI assistant, able to provide highly personalized and useful answers to specific users' questions based on information they've given you. Your answer should be personal and tailored to the users' preferences and personality, but should also directly answer the question and be generally helpful. Their question may be one that is well-suited to personalized answers, or may be a more general objective or fact-based question. Regardless, your answer should be as helpful as possible to the user below. Your should incorporate as much relevant personal information they've given you as possible to as much as possible when constructing your helpful answer, such that the answer feels tailored to them. Whenever possible, tie your answer back to their background in order to ensure the answer is relevant or helpful to them. Feel free to use their name if it is given to you. The user's initial question is given below:"""

]

def extract_history(
    conversation: str,
    ) -> str:
    conversation = conversation[conversation.find('The initial request is as follows: ') + len('The initial request is as follows: '):]
    conversation = strip_whitespace_around_substring(conversation, B_INST)
    conversation = strip_whitespace_around_substring(conversation, E_INST)
    conversation = strip_whitespace_around_substring(conversation, BOS_TOKEN)
    conversation = strip_whitespace_around_substring(conversation, EOS_TOKEN)
    return conversation


def create_turns(
    conversation: str
    ) -> List[str]:
    delim_1 = EOS_TOKEN + B_INST
    delim_2 = E_INST

    # Escape the delimiters to make them safe for use in a regex pattern
    escaped_delim_1 = re.escape(delim_1)
    escaped_delim_2 = re.escape(delim_2)

    # Create a regex pattern that matches either delimiter
    pattern = f"{escaped_delim_1}|{escaped_delim_2}"
    turns = re.split(pattern, conversation)
    # Note that because of the final E_INST token, there is an additional empty stirng at the end of the list
    # As a result, we return everything before the last 1 elements
    return turns[:-1]



def strip_whitespace_around_substring(s, substring):
    # The pattern looks for the substring followed by any amount of whitespace (\s*)
    # and replaces it with just the substring.
    pattern = r'\s*' + re.escape(substring) + r'\s*'
    return re.sub(pattern, substring, s)


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
