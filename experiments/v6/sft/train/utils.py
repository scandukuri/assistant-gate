from typing import Dict, Sequence, List, Tuple
import torch
import io
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
        

def _tokenize_fn(
    messages: Sequence[Dict], 
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Tokenize a list of strings. Edited from from https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py."""
    
    tokenized_list = [
        tokenizer(
            dct['content'],
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
            add_special_tokens=False,
        )
        for dct in messages
    ]
    input_id_list = [tokenized.input_ids[0] for tokenized in tokenized_list]
    label_list = [label.fill_(IGNORE_INDEX) if i % 2 == 0 else label for i, label in enumerate(input_id_list.deepcopy())]
    # Assuming input_id_list and label_list are your lists of 1D tensors
    input_ids = torch.cat(input_id_list, dim=0)
    labels = torch.cat(label_list, dim=0)
    

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
    tokenized = [_tokenize_fn(messages, tokenizer) for messages in targets]
    
    
    ### TODO: STILL NEED TO IMPLEMENT THE LOGIC FOR GETTING input_ids, labels into a single format for the Dataset
    ### atm, targets_tokenized["input_ids"] and targets_tokenized["labels"] are dicts with a single tensor representing a single example
    final_dict = {key: [d[key] for d in tokenized] for key in tokenized[0]}
        
    train_dataset = Dataset.from_dict(final_dict)
    train_dataset.set_format('torch')
    return train_dataset


