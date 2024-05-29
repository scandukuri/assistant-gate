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
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp.api import FullStateDictConfig, FullOptimStateDictConfig


# TRANSFORMERS IMPORTS
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_scheduler
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from datasets import Dataset


# CUSTOM AND LOCAL IMPORTS
from paths import *
from global_utils import *
from AG.trainers.SFTTrainerUtils import *


class SFTTrainer:
    def __init__(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        config: DictConfig,
        train_dataset: List[Dict],
        eval_dataset: List[Dict],
        local_rank: int,
        world_size: int,
        save_option: str,
    ):
        """Intialize the TYPOTrainer.

        Args:
            model: A transformers.PreTrainedModel.
            tokenizer: A transformers.PreTrainedTokenizer.
            train_dataset: List of training examples.
            eval_dataset: List of evaluation examples.
            config: Training configuration.
            local_rank: the rank for distributed training
            world_size: num gpus
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.local_rank = local_rank
        self.world_size = world_size
        self.save_option = save_option

        

        train_dataset = CustomSFTDataset(train_dataset)
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.training.train_batch_size,
            collate_fn=CustomSFTDataCollator(tokenizer),
            shuffle=False,
            sampler=train_sampler
        )


        # optimizer
        if self.save_option == "hf":
            self.optimizer = AdamW(model.parameters(), lr=config.training.lr)

        elif self.save_option == "pt":
            self.optimizer = RMSprop(model.parameters(), lr=config.training.lr)

        # scheduler
        self.scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=config.training.num_warmup_steps,
            num_training_steps=(
                len(self.train_dataloader) * self.config.training.n_epochs
            )
            // config.training.gradient_accumulation_steps,
        )

        # writing checkpoints
        self.checkpoint_dir = config.training.checkpoint_dir
        print("Loaded model on rank", self.local_rank)
        print("Loaded reference model on rank", self.local_rank)
        print(f"Writing checkpoints to {self.config.training.checkpoint_dir}.")
        dist.barrier()

    def save_model(
        self,
        epoch: int,
        state: Dict[str, torch.Tensor],
    ):
        """Merges checkpoint with HF model and writes to dir."""
        checkpoint_dir = os.path.join(
            self.config.training.checkpoint_dir, f"epoch-{epoch}"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

        if self.save_option == "hf":  # this is prob not necessary...
            save_model = AutoModelForCausalLM.from_pretrained(
                **self.config.model_config,
            )
            save_model.load_state_dict(state)
            save_model.save_pretrained(checkpoint_dir)
            self.tokenizer.save_pretrained(checkpoint_dir)

            print("Model saved, deleting model...")
            del save_model
            print("Deleted model...")

        elif self.save_option == "pt":
            torch.save(state, os.path.join(checkpoint_dir, "model.pt"))

    def save_checkpoint(self, epoch):
        """
        Save model, gathering from all processes
        and saving only on the rank 0 process. Copy-pasted from dpo repo.
        """
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(
            self.model, StateDictType.FULL_STATE_DICT, state_dict_config=save_policy
        ):
            model_state_dict = self.model.state_dict()
            if self.local_rank == 0:
                self.save_model(
                    epoch=epoch,
                    state=model_state_dict,
                )
            del model_state_dict
        dist.barrier()

    def compute_metrics(
        self,
        model: transformers.PreTrainedModel,
        reference_model: transformers.PreTrainedModel,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Computes metrics for both policy and reference model, for EITHER 'chosen' or 'rejected' sequences (but not both)"""

        # get logprobs for policy model
        logits, labels = prepare_logits_labels(
            model, batch["input_ids"], batch["labels"], batch['attention_mask']
        )

        policy_batch_logprobs = get_batch_logprobs(logits, labels)


        # get logprobs for reference model
        with torch.no_grad():
            ref_logits, ref_labels = prepare_logits_labels(
                reference_model, batch["input_ids"], batch["labels"], batch['attention_mask']
            )
            reference_batch_logprobs = get_batch_logprobs(ref_logits, ref_labels)

        return policy_batch_logprobs, reference_batch_logprobs
        # new return should match the expected return for DPO repository


    def _run_batch(self, batch: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        """Run a batch and compute the standard SFT loss."""
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        loss = outputs.loss
        return loss


    def train(self):
        print(f"Training started on Rank {self.local_rank}.")
        
        for epoch in range(self.config.training.n_epochs):
            print(f"Rank {self.local_rank}, Epoch {epoch}: Start.")
            accumulated_loss = 0
            for step, batch in tqdm.tqdm(
                enumerate(self.train_dataloader), desc=f"Epoch {epoch}"
            ):
                if self.local_rank == 0:
                    print(f"Rank {self.local_rank}, Step {step}: Processing batch")
                    
                batch = {k: v.to(self.local_rank) for k, v in batch.items()}
                
                loss = self._run_batch(batch)
                loss.backward()
                
                if step % self.config.training.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                loss_value = loss.item()
                accumulated_loss += loss_value
                
                if self.local_rank == 0:
                    print(f"Rank {self.local_rank}, Step {step}: Loss = {loss_value}")
                    wandb.log({
                        "train/loss": loss_value,
                    })

            # Log epoch metrics and save checkpoint
            if self.local_rank == 0:
                wandb.log({
                    "train/epoch_loss": accumulated_loss / len(self.train_dataloader),
                })
            self.save_checkpoint(epoch)
        
        print(f"Training completed on Rank {self.local_rank}.")





class CustomSFTDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# @dataclass
# class CustomSFTDataCollator:
#     tokenizer: transformers.PreTrainedTokenizer

#     def collate_fn(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
#         """
#         Helper function to collate lists of instances. This function pads each tensor to the maximum length found in the batch.
#         """
#         collated_batch = {}
#         unique_keys = instances[0].keys() if instances else []
#         max_length = max(
#             (len(instance[key]) for instance in instances for key in unique_keys),
#             default=0
#         )

#         for key in unique_keys:
#             values = [torch.tensor(instance[key], dtype=torch.long) if not isinstance(instance[key], torch.Tensor)
#                       else instance[key] for instance in instances]
#             padding_value = self.tokenizer.pad_token_id if 'input_ids' in key else -100 if 'labels' in key else 0
#             padded_values = [torch.nn.functional.pad(value, (0, max_length - value.size(0)), value=padding_value)
#                              for value in values]
            
#             collated_batch[key] = torch.stack(padded_values)
#         return collated_batch




@dataclass
class CustomSFTDataCollator:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Args:
            instances: A list of tuples, each tuple contains two dictionaries ('chosen' and 'rejected')
                       with keys for input_ids, attention_mask, and optionally labels. Each dictionary represents a tokenized example.
        """
        result = self._collate(instances)
        return result

    def _collate(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """
        Helper function to collate lists of instances. This function pads each tensor to the maximum length found in the batch.
        """
        collated_batch = {}
        unique_keys = instances[0].keys() if instances else []
        max_length = max(
            (len(instance[key]) for instance in instances for key in unique_keys),
            default=0
        )

        for key in unique_keys:
            values = [torch.tensor(instance[key], dtype=torch.long) if not isinstance(instance[key], torch.Tensor)
                      else instance[key] for instance in instances]
            padding_value = self.tokenizer.pad_token_id if 'input_ids' in key else -100 if 'labels' in key else 0
            padded_values = [torch.nn.functional.pad(value, (0, max_length - value.size(0)), value=padding_value)
                             for value in values]
            
            collated_batch[key] = torch.stack(padded_values)
        return collated_batch