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
    CheckpointImpl,
)

from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import torch.nn.functional as F


# TRANSFORMERS IMPORTS
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_scheduler,
)
from transformers import TrainingArguments, Trainer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from datasets import Dataset


# CUSTOM AND LOCAL IMPORTS
from paths import *
from utils import *
from global_utils import *
from AG.trainers.SFTTrainer import *
from AG.trainers.SFTTrainerUtils import *



######
# FSDP UTILITIES
######

def get_policy(blocks={LlamaDecoderLayer}):
    """Wrapping policy setup."""
    return functools.partial(
        transformer_auto_wrap_policy, transformer_layer_cls=blocks
    )

    
def init_distributed(
    rank: int,
    world_size: int,
    master_addr: str = "localhost",
    port: int = 12355,
    backend: str = "nccl",
):
    print(rank, "initializing distributed")
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def worker_main(rank: int, world_size: int, args: DictConfig, model):
    init_distributed(rank, world_size)
    if rank == 0:
        print("Initialized process group...")
        print("WANDB")
        wandb.init(project=args.model.wandb.project, name=args.model.wandb.name)

    # # get tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(**args.model.tokenizer_config)
    
    # set padding
    tokenizer.pad_token, tokenizer.padding_side = tokenizer.eos_token, "right"

    # fsdp 
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    print(f"wrapping model {rank}...")
    
    non_reentrant_wrapper = functools.partial(
        checkpoint_wrapper,
        # offload_to_cpu=False,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    
    model = FSDP(
        model,
        auto_wrap_policy=get_policy(),
        mixed_precision=mp_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=False,
    )

    check_fn = lambda submodule: isinstance(submodule, LlamaDecoderLayer)
    apply_activation_checkpointing(model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)
    
    print(f"wrapped model {rank}...")

    # training args
    if not os.path.exists(args.qa_model.training_args.training.checkpoint_dir):
        os.makedirs(args.qa_model.training_args.training.checkpoint_dir)
    if not os.path.exists(f'{args.qa_model.training_args.training.checkpoint_dir}/final'):
        os.makedirs(f'{args.qa_model.training_args.training.checkpoint_dir}/final')

    targets = json.load(open(f"{SFT_DATA_PATH}/{LLAMA_VERSION}/{args.qa_model.shortname}/{args.split.name}.json", 'r'))[:200]
    dataset = sft_preprocess(targets=targets, tokenizer=tokenizer)
    print("Is it a dataset ", isinstance(dataset, Dataset))
    # dataset = dataset.shuffle(seed=42)
    
    
    # dataset = dataset.train_test_split(test_size=args.validation_split_size)
    logging.info(dataset)
   
    # collator for sft
    data_collator = CustomSFTDataCollator(tokenizer=tokenizer)
    
    if rank == 0:
        print(len(targets))
        print(f"tokenized {len(targets)} training examples...")


    # Assuming both datasets are the same size and properly aligned
    dataset_size = len(dataset)  # Both datasets should have the same number of elements
    eval_size = int(args.validation_split_size * dataset_size)  # 2.5% of the dataset
    
    # Randomly select indices for the evaluation set
    eval_indices = random.sample(range(dataset_size), eval_size)
    # Create evaluation and training datasets
    eval_dataset = [dataset[i] for i in eval_indices]
    train_dataset = [dataset[i] for i in range(dataset_size) if i not in eval_indices]
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        config=args.qa_model.training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        local_rank=rank,
        world_size=world_size,
        save_option="hf",
    )
    
    trainer.train()

# main training script 
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args: DictConfig) -> None:
    random.seed(1)
    logging.info(f"Training {args.model.shortname} on {args.split.name} split...")
    logging.info(f"Writing checkpoints to: {args.qa_model.training_args.training.checkpoint_dir}")
    logging.info(f"Wandb name: {args.model.wandb.name}")
    logging.info(f"Max seq length: {args.model.tokenizer_config.model_max_length}")
    

    # nans 
    torch.autograd.set_detect_anomaly(True)


    # seeds
    torch.cuda.manual_seed(args.qa_model.training_args.training.seed)
    torch.manual_seed(args.qa_model.training_args.training.seed)
    random.seed(args.qa_model.training_args.training.seed)


    # get model
    model = AutoModelForCausalLM.from_pretrained(**args.model.model_config, torch_dtype=torch.bfloat16)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})

    # run with resource limits
    world_size = torch.cuda.device_count()
    print("WORLD SIZE", world_size,"MODEL", args.model.model_config)
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    print(f'setting RLIMIT_NOFILE soft limit to {hard} from {soft}')
    mp.spawn(worker_main, nprocs=world_size, args=(world_size, args, model))


if __name__ == "__main__":
    main()