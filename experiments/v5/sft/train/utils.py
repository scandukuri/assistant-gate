from typing import Dict, Sequence, List, Tuple
import torch
import io
import os
import json
import copy
from dataclasses import dataclass

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
    strings: Sequence[str], 
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Tokenize a list of strings. Copy-pasted from https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
            add_special_tokens=False,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: List[str],
    targets: List[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dataset:
    """Preprocess the data by tokenizing."""
    
    targets_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (targets, sources)]
    input_ids = targets_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)

    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
        
    train_dataset = Dataset.from_dict(dict(input_ids=input_ids, labels=labels))
    train_dataset.set_format('torch')
    return train_dataset


