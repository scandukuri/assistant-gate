import pandas as pd
import json
import logging
import hydra
from omegaconf import DictConfig
import argparse
import fire
import datasets
from datasets import load_dataset

PUBLIC_ROWS = 1000
PRIVATE_ROWS = 100


logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="conf", config_name='config')
def main(args: DictConfig) -> None:
    logging.info("Extracting initial prompts...")
    
    def extract_initial_prompt(
        conversation: str
        ) -> str:
        return conversation[:conversation.find('Assistant: ')].strip()[len('Human: '):]
        

    full_dataset = load_dataset("Dahoas/instruct-human-assistant-prompt", split='train').to_pandas()
    public_exchanges = full_dataset.iloc[:PUBLIC_ROWS]['prompt'].tolist()
    public_initial_turns = [extract_initial_prompt(exchange) for exchange in public_exchanges]
    
    private_exchanges = full_dataset.iloc[PUBLIC_ROWS:PUBLIC_ROWS + PRIVATE_ROWS]['prompt'].tolist()
    private_initial_turns = [extract_initial_prompt(exchange) for exchange in private_exchanges]
    
    with open(f'public/prompts.json', 'w') as f:
        json.dump(public_initial_turns, f)
    
    with open(f'private/prompts.json', 'w') as f:
        json.dump(private_initial_turns, f)
    
    

if __name__ == '__main__':
    try:
        fire.Fire(main())
    except:
        pass