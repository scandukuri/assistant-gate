import pandas as pd
import json
import logging
import hydra
from omegaconf import DictConfig
import argparse
import fire
import datasets
from datasets import load_dataset


N_ROWS = 5


logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="../conf", config_name='config')
def main(args: DictConfig) -> None:
    logging.info("Extracting initial prompts...")
    
    def extract_initial_prompt(
        conversation: str
        ) -> str:
        return conversation[:conversation.find('Assistant: ')].strip()[len('Human: '):]
        

    full_dataset = load_dataset("Dahoas/instruct-human-assistant-prompt", split='train').to_pandas()
    chosen_exchanges = full_dataset.iloc[:N_ROWS]['prompt'].tolist()
    initial_turns = [extract_initial_prompt(exchange) for exchange in chosen_exchanges]
    
    with open(f'instruct-questions/first-{N_ROWS}-prompts.json', 'w') as f:
        json.dump(initial_turns, f)
    
    
    

if __name__ == '__main__':
    try:
        fire.Fire(main())
    except:
        pass