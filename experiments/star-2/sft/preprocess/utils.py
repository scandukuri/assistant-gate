from typing import List, Tuple
import re


B_INST, E_INST, BOS_TOKEN, EOS_TOKEN = '[INST]', '[/INST]', '<s>', '</s>'


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