from typing import Dict, Sequence, List, Tuple
import torch
import io
import logging
import os
import json
import copy
from dataclasses import dataclass
from typing import Dict, List, Sequence

import transformers
from omegaconf import DictConfig
from datasets import Dataset, load_dataset

IGNORE_INDEX = -100
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning. Copy-pasted from https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        

# Function to find sequence in tensor
def find_sequence(tensor, sequence):
    seq_len = sequence.size(0)
    indices = []

    for i in range(tensor.size(0) - seq_len + 1):
        if torch.equal(tensor[i:i+seq_len], sequence):
            indices.append(i)

    return indices

# Function to create a mask for L, masking out occurrences between and including sequences a and b
def create_mask(L, a, b):
    # Initial mask with all True
    mask = torch.ones(L.size(0), dtype=torch.bool)
    
    # Find start indices of sequence a
    start_indices_a = find_sequence(L, a)
    
    for start_index_a in start_indices_a:
        # For each start of a, find the start of b after a
        start_index_b = find_sequence(L[start_index_a:], b)
        
        if start_index_b:
            # Assuming we mask from the first occurrence of b after a
            start_index_b = start_index_b[0] + start_index_a  # Adjust index relative to the whole tensor
            end_index_b = start_index_b + b.size(0)  # End index of b
            
            # Set mask to False from start of a to end of b
            mask[start_index_a:end_index_b] = False
    
    return mask

def _tokenize_fn(
    messages: Sequence[Dict], 
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Tokenize a list of strings. Edited from from https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py."""
    
    enumerated = list()
    for i in range(len(messages) + 1):
        enumerated.append(tokenizer.apply_chat_template(messages[:i], return_tensors='pt', padding='longest', max_length=tokenizer.model_max_length, truncation=True))
    enumerated = [e[0] for e in enumerated][1:]
    input_ids = copy.deepcopy(enumerated[-1])
    
    
    # Mask user interactions
    for i in range(1, len(enumerated), 2):
        if i == 1:
            enumerated[-1][0 : len(enumerated[i - 1])] = IGNORE_INDEX
        else:
            enumerated[-1][len(enumerated[i - 2]): len(enumerated[i - 1])] = IGNORE_INDEX
    labels = copy.deepcopy(enumerated[-1])
    
    return dict(
        input_ids=input_ids,
        labels=labels,
    )


def preprocess(
    targets: List[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dataset:
    """Preprocess the data by tokenizing."""
    # targets is a list of messages in the form of 'user's message', 'system's response', 'user's message', 'system's response', ...
    # reference split-conversations.py in preprocessing
    tokenized = list()
    for i, messages in enumerate(targets):
        if i % 500 == 0: logging.info(f'Processed {i} targets out of {len(targets)}')
        tokenized.append(_tokenize_fn(messages, tokenizer))
    
    
    ### TODO: STILL NEED TO IMPLEMENT THE LOGIC FOR GETTING input_ids, labels into a single format for the Dataset
    ### atm, targets_tokenized["input_ids"] and targets_tokenized["labels"] are dicts with a single tensor representing a single example
    final_dict = {key: [d[key] for d in tokenized] for key in tokenized[0]}
    
    train_dataset = Dataset.from_dict(final_dict)
    train_dataset.set_format('torch')
    return train_dataset


