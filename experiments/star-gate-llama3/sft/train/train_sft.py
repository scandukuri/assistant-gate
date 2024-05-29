import os
import logging 
from tqdm import tqdm

import random
import fire
import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

from utils import *
from paths import *

logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args: DictConfig) -> None:
    random.seed(1)
    logging.info(f"Training {args.qa_model.shortname} on {args.split.name} split...")
    logging.info(f"Writing checkpoints to: {args.qa_model.training_args.output_dir}")
    logging.info(f"Wandb name: {args.qa_model.wandb.name}")
    logging.info(f"Max seq length: {args.qa_model.tokenizer_config.model_max_length}")
    logging.info(f"Devices: {torch.cuda.device_count()}")
    
    # wandb
    args_dict = OmegaConf.to_container(args, resolve=True)
    wandb.init(project=args.qa_model.wandb.project, name=args.qa_model.wandb.name, config=args_dict)


    accelerator = Accelerator()
    # get tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(**args.model.tokenizer_config)
    
    # set padding
    tokenizer.pad_token, tokenizer.padding_side = tokenizer.eos_token, "right"
   
    # get model
    model = AutoModelForCausalLM.from_pretrained(
        **args.model.model_config, 
        torch_dtype=torch.bfloat16,
        device_map='auto',
    )
    
    # training args
    training_args_dict = OmegaConf.to_container(args.qa_model.training_args, resolve=True)
    training_args = TrainingArguments(**training_args_dict)
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    if not os.path.exists(f'{training_args.output_dir}/final'):
        os.makedirs(f'{training_args.output_dir}/final')

    targets = json.load(open(f"{SFT_DATA_PATH}/{LLAMA_VERSION}/{args.qa_model.shortname}/{args.split.name}.json", 'r'))
    dataset = preprocess(targets=targets, tokenizer=tokenizer)
    dataset = dataset.shuffle(seed=42)
    


    dataset = dataset.train_test_split(test_size=args.validation_split_size)
    logging.info(dataset)
   
    # collator for sft
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    # trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer, 
        args=training_args, 
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        data_collator=data_collator,
    )
    
    # train
    trainer.train()
    
    # save trainer
    trainer.save_model(output_dir=f'{training_args.output_dir}/final')
    
    # save pretrained 
    trainer.model.save_pretrained(f'{training_args.output_dir}/final')
    
if __name__ == "__main__":
    fire.Fire(main())
