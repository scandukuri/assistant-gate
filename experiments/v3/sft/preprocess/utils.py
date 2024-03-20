from typing import List, Tuple
import re

CONVERSATIONS_DIR = '/sailhome/andukuri/research_projects/assistant-gate/experiments/v3/conditions/qa-experimental/simulated-conversations/public/qa-model-mistral-7b-instruct-v02-vllm_human-model-mistral-7b-instruct-v02-vllm_qa-12_humansys-1_human-4_maxturns-4_top-k-1_conversations.json'

B_INST, E_INST, BOS_TOKEN, EOS_TOKEN = '[INST]', '[/INST]', '<s>', '</s>'
ROLES = ["user", "assistant"]

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
    # In addition, the last turn is always the human response to the qa model's last question (we don't need the human response either)
    # As a result, we return everything before the last 2 elements
    return turns[:-2]



def create_examples(
    conversation: str,
    ) -> List[Tuple]:
    e_inst_indices = [m.start() for m in re.finditer(re.escape(E_INST), conversation)][:-1]
    eos_indices = [m.start() for m in re.finditer(re.escape(EOS_TOKEN), conversation)]
    zipped_examples = [(conversation[:a + len(E_INST)].strip(), conversation[:b + len(EOS_TOKEN)].strip()) for (a, b) in zip(e_inst_indices, eos_indices)]
    
    # unzips a list of n pair tuples into a list of two length n tuples that are index-wise corresponding pairs
    unzipped = list(zip(*zipped_examples))
    return list(unzipped[0]), list(unzipped[1])


def strip_whitespace_around_substring(s, substring):
    # The pattern looks for the substring followed by any amount of whitespace (\s*)
    # and replaces it with just the substring.
    pattern = r'\s*' + re.escape(substring) + r'\s*'
    return re.sub(pattern, substring, s)