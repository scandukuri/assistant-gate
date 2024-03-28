from typing import List, Tuple
import re
import logging


B_INST, E_INST, BOS_TOKEN, EOS_TOKEN = '[INST]', '[/INST]', '<s>', '</s>'

RATER_SYS_PROMPTS = [
    # 0
    """You are an expert in evaluating assistant responses, particularly skilled in analyzing their alignment with user backgrounds and preferences."""
]
RATER_SYS_PROMPT_IDX = 0

RATER_MAIN_PROMPTS = [
    # 0
    """For the following user query to an assistant, which of the assistant responses is more aligned with the user's background and preferences?

User Background: {}

User Query: {}

Assistant Response A: {}

Assistant Response B: {}

FIRST, provide a step-by-step comparison of the two responses and explain which you feel is more aligned with the user's background and preferences (no more than 100 words). 

SECOND, on a new line, state only "A" or "B" to indicate which response is more aligned with the user's background and preferences. 

Comparison: <step-by-step comparison and explanation>

Final Response: <"A" or "B">""",
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


