# GENERAL IMPORTS
from typing import Dict, List, Optional, Tuple, Sequence
from dataclasses import dataclass
from datetime import timedelta
import os
import json
import random
import functools
import resource
import logging
import copy
import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb


# TORCH IMPORTS
import torch
from torch import nn, multiprocessing as mp, distributed as dist
from torch.optim import AdamW, RMSprop
from torch.utils.data import DataLoader, distributed as dist_data
from torch.distributed import fsdp
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    StateDictType,
    api as fsdp_api
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    apply_activation_checkpointing,
    CheckpointImpl
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import torch.nn.functional as F


# TRANSFORMERS IMPORTS
import transformers
from transformers import (
    PreTrainedTokenizer,
    AutoTokenizer,
    AutoModelForCausalLM,
    get_scheduler
)
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer 
from datasets import Dataset


# CUSTOM AND LOCAL IMPORTS
from paths import *
from global_utils import *


IGNORE_INDEX = -100
PLACEHOLDER_STRING = "Model failed to provide a preference."

######
# CHINMAYA / PHILIPP UTILITIES
######


def dpo_loss(
    beta: float,
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.
    Extended from https://github.com/github-copilot/code_referencing?cursor=2e0f641eb0ed94f4be74d4ea5b0dcec6&editor=vscode
    
    Args:
        beta: The temperature parameter for the DPO loss.
        label_smoothing: The label smoothing parameter for the DPO loss.
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    if torch.cuda.current_device() == 0:
        print(f"pi_logratios: {pi_logratios}")
        print(f"ref_logratios: {ref_logratios}")
        print(f"pi_logratios - ref_logratios: {pi_logratios - ref_logratios}")
        print(f"policy_chosen_logps: {policy_chosen_logps}")
        print(f"reference_chosen_logps: {reference_chosen_logps}")
        print(f"policy_rejected_logps: {policy_rejected_logps}")
        print(f"reference_rejected_logps: {reference_rejected_logps}")


    logits = pi_logratios - ref_logratios

    losses = -F.logsigmoid(beta * logits)
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
    return losses, chosen_rewards, rejected_rewards


def get_batch_logprobs(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    ignore_idx: int = -100,
) -> torch.FloatTensor:
    """Computes the log probabilities of labels given logits.

    Args:
        logits: The logits of the model. Shape: (batch_size, sequence_length, vocab_size).
        labels: The labels of the model. Shape: (batch_size, sequence_length).
        ignore_idx: The index to ignore in the loss computation; defaults to -100.

    Returns:
        average_logprobs: The log probabilities of the labels. Shape: (batch_size, ).
    """
    assert (
        logits.shape[:-1] == labels.shape
    ), "Logits and labels must have the same shape."

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1]
    loss_mask = labels == ignore_idx

    per_token_logprobs = -F.cross_entropy(
        input=logits.reshape(-1, logits.size(-1)),
        target=labels.reshape(-1),
        reduction="none",
    ).reshape(labels.shape)

    per_token_logprobs[loss_mask] = 0

    # average token logprobs
    average_token_logprobs = per_token_logprobs.sum(dim=-1) / torch.logical_not(
        loss_mask
    ).sum(dim=1)

    return average_token_logprobs


def prepare_logits_labels(
    model: torch.nn.Module,
    input_ids,
    labels,  # the labels with the desired USER turns masked out
    attention_masks,
    ignore_idx: Optional[int] = -100,
) -> Tuple[torch.FloatTensor, torch.LongTensor]:
    """Prepares the logits and labels for the given prompts and responses.

    Args:
        model: A torch.nn.Module model.
        batch: A batch of tokenized examples. Each value should be a tensor of shape (batch_size, sequence_length).
        labels: (batch_size, sequence_length), but all of the USER tokens are already set to IGNORE_INDEX
        ignore_index
    Returns:
        logits: The logits of the model. Shape: (batch_size, sequence_length, vocab_size).
        labels: The labels of the model. Shape: (batch_size, sequence_length).
    """
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_masks,
    ).logits

    return logits, labels.to(model.device)


# Function to find sequence in tensor
def find_sequence(tensor, sequence):
    seq_len = sequence.size(0)
    indices = []

    for i in range(tensor.size(0) - seq_len + 1):
        if torch.equal(tensor[i : i + seq_len], sequence):
            indices.append(i)

    return indices


def _tokenize_fn(
    messages: Sequence[Dict],
    tokenizer: transformers.PreTrainedTokenizer,
    IGNORE_INDEX: Optional[int] = -100,
) -> Dict:
    """Tokenize a list of strings. Edited from from https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py."""

    enumerated = list()
    for i in range(1, len(messages) + 1):
        enumerated.append(
            tokenizer.apply_chat_template(
                messages[:i],
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            )
        )
    enumerated = [e[0] for e in enumerated]
    input_ids = copy.deepcopy(enumerated[-1])

    # Mask user interactions
    for i in range(1, len(enumerated), 2):
        if i == 1:
            enumerated[-1][0 : len(enumerated[i - 1])] = IGNORE_INDEX
        else:
            enumerated[-1][
                len(enumerated[i - 2]) : len(enumerated[i - 1])
            ] = IGNORE_INDEX
    labels = copy.deepcopy(enumerated[-1])

    return dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=torch.ones_like(input_ids),
    )


def sft_preprocess(
    targets: List[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dataset:
    """Preprocess the data by tokenizing."""
    # targets is a list of messages in the form of 'user's message', 'system's response', 'user's message', 'system's response', ...
    # reference split-conversations.py in preprocessing
    tokenized = list()
    print("Starting preprocessing...")
    print(f"Test: {targets[0]}")
    for i, messages in enumerate(targets):
        if i % 50 == 0:
            print(f"Processed {i} targets out of {len(targets)}")
        tokenized.append(_tokenize_fn(messages, tokenizer))
    print("Post-tokenization...", "\n\n",tokenized[0])
    # final_dict = {key: [d[key] for d in tokenized] for key in tokenized[0]}

    # train_dataset = Dataset.from_dict(final_dict)
    # train_dataset.set_format('torch')
    # return train_dataset
    return tokenized





######
# ERIC MITCHELL UTILITIES
######

import os
import getpass
from datetime import datetime
import torch
import random
import numpy as np
import torch.distributed as dist
import inspect
import importlib.util
import socket
import os
from typing import Dict, Union, Type, List


def get_open_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0)) # bind to all interfaces and use an OS provided port
        return s.getsockname()[1] # return only the port number


def get_remote_file(remote_path, local_path=None):
    hostname, path = remote_path.split(':')
    local_hostname = socket.gethostname()
    if hostname == local_hostname or hostname == local_hostname[:local_hostname.find('.')]:
        return path
    
    if local_path is None:
        local_path = path
    # local_path = local_path.replace('/scr-ssd', '/scr')    
    if os.path.exists(local_path):
        return local_path
    local_dir = os.path.dirname(local_path)
    os.makedirs(local_dir, exist_ok=True)

    print(f'Copying {hostname}:{path} to {local_path}')
    os.system(f'scp {remote_path} {local_path}')
    return local_path


def rank0_print(*args, **kwargs):
    """Print, but only on rank 0."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)


def get_local_dir(prefixes_to_resolve: List[str]) -> str:
    """Return the path to the cache directory for this user."""
    for prefix in prefixes_to_resolve:
        if os.path.exists(prefix):
            return f"{prefix}/{getpass.getuser()}"
    os.makedirs(prefix)
    return f"{prefix}/{getpass.getuser()}"
    

def get_local_run_dir(exp_name: str, local_dirs: List[str]) -> str:
    """Create a local directory to store outputs for this run, and return its path."""
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S_%f")
    run_dir = f"{get_local_dir(local_dirs)}/{exp_name}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def slice_and_move_batch_for_device(batch: Dict, rank: int, world_size: int, device: str) -> Dict:
    """Slice a batch into chunks, and move each chunk to the specified device."""
    chunk_size = len(list(batch.values())[0]) // world_size
    start = chunk_size * rank
    end = chunk_size * (rank + 1)
    sliced = {k: v[start:end] for k, v in batch.items()}
    on_device = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in sliced.items()}
    return on_device


def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat([tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim)


def all_gather_if_needed(values: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """Gather and stack/cat values from all processes, if there are multiple processes."""
    if world_size == 1:
        return values

    all_values = [torch.empty_like(values).to(rank) for _ in range(world_size)]
    dist.all_gather(all_values, values)
    cat_function = torch.cat if values.dim() > 0 else torch.stack
    return cat_function(all_values, dim=0)


def formatted_dict(d: Dict) -> Dict:
    """Format a dictionary for printing."""
    return {k: (f"{v:.5g}" if type(v) == float else v) for k, v in d.items()}
    

def disable_dropout(model: torch.nn.Module):
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def print_gpu_memory(rank: int = None, message: str = ''):
    """Print the amount of GPU memory currently allocated for each GPU."""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            device = torch.device(f'cuda:{i}')
            allocated_bytes = torch.cuda.memory_allocated(device)
            if allocated_bytes == 0:
                continue
            print('*' * 40)
            print(f'[{message} rank {rank} ] GPU {i}: {allocated_bytes / 1024**2:.2f} MB')
        print('*' * 40)


def get_block_class_from_model(model: torch.nn.Module, block_class_name: str) -> torch.nn.Module:
    """Get the class of a block from a model, using the block's class name."""
    for module in model.modules():
        if module.__class__.__name__ == block_class_name:
            return module.__class__
    raise ValueError(f"Could not find block class {block_class_name} in model {model}")


def get_block_class_from_model_class_and_block_name(model_class: Type, block_class_name: str) -> Type:
    filepath = inspect.getfile(model_class)
    assert filepath.endswith('.py'), f"Expected a .py file, got {filepath}"
    assert os.path.exists(filepath), f"File {filepath} does not exist"
    assert "transformers" in filepath, f"Expected a transformers model, got {filepath}"

    module_name = filepath[filepath.find('transformers'):].replace('/', '.')[:-3]
    print(f"Searching in file {filepath}, module {module_name} for class {block_class_name}")

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get the class dynamically
    class_ = getattr(module, block_class_name)
    print(f"Found class {class_} in module {module_name}")
    return class_


def init_distributed(rank: int, world_size: int, master_addr: str = 'localhost', port: int = 12355, backend: str = 'nccl'):
    print(rank, 'initializing distributed')
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


class TemporarilySeededRandom:
    def __init__(self, seed):
        """Temporarily set the random seed, and then restore it when exiting the context."""
        self.seed = seed
        self.stored_state = None
        self.stored_np_state = None

    def __enter__(self):
        # Store the current random state
        self.stored_state = random.getstate()
        self.stored_np_state = np.random.get_state()

        # Set the random seed
        random.seed(self.seed)
        np.random.seed(self.seed)

    def __exit__(self, exc_type, exc_value, traceback):
        # Restore the random state
        random.setstate(self.stored_state)
        np.random.set_state(self.stored_np_state)