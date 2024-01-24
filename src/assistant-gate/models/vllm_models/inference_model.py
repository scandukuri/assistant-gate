import os
from typing import Optional, List, Tuple

import random
import torch
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams


import logging 


logging.basicConfig(level=logging.INFO)


class VLLMInferenceModel():
    """Wrapper for running inference with VLLM."""
    def __init__(
        self, 
        model: str = "mistralai/Mistral-7B-Instruct-v0.1",
        download_dir: str = "/scr/andukuri/printllama/Mixtral-8x7B-v0.1",
        dtype: str = "auto",
        tensor_parallel_size: int = 1,
        quantization: Optional[str] = None,
        seed: int = 1,
    ):
        """Initializes VLLM Inference Model"""
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model,
            cache_dir='scr/andukuri/printllama/',
            token=os.getenv("HF_TOKEN"),
        )
        
        self.tokenizer.pad_token = "[PAD]"
        self.tokenizer.padding_side = "right"

        #random.seed(seed)
        print("Loaded tokenizer")
        self.model = LLM(
            model=model,
            download_dir=download_dir,
            dtype=torch.float16 if quantization == "awq" else dtype,
            tensor_parallel_size=tensor_parallel_size,
            quantization=quantization,
        )
        
        
    @property
    def model_type(self):
        return "VLLMInferenceModel"
    
    
    def batch_log_probs(
        self, 
        answers: List[str], 
        prompts: List[str],
    ) -> torch.Tensor:
        """Returns log probabilities of prompts including answer (answers) and prompts excluding answers (prompts)."""
        # TOKENIZE
        with torch.no_grad():
            tokenized_answers = self.tokenizer(
                answers,
                add_special_tokens=False,
                return_tensors="pt",
                padding=True,
            )
                    
            tokenized_prompts = self.tokenizer(
                prompts,
                add_special_tokens=False,
                return_tensors="pt",
                padding="max_length",
                max_length=tokenized_answers.input_ids.shape[1],
            )
            
            tokenized_answers_input_ids = tokenized_answers.input_ids.tolist()
              
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=1,
                n=1,
                prompt_logprobs=0,
                spaces_between_special_tokens=False,
            )
                        
            output_answers = self.model.generate(
                prompt_token_ids=tokenized_answers_input_ids,
                sampling_params=sampling_params,
                use_tqdm=True,
            )
  
            
            # now get the tokens back
            log_probs_answers = torch.tensor([
                [v for prob in output_answer.prompt_logprobs[1:] for _, v in prob.items()]
                for output_answer in output_answers
            ])
            
            # MASK FOR FINAL ANSWERS 
            labels = tokenized_answers.input_ids[:, 1:]
            mask = torch.logical_and(tokenized_prompts.input_ids[:, 1:] == 0, labels != 0)
            log_probs_answers.masked_fill_(~mask, 0) 
            log_probs = log_probs_answers.sum(dim=-1)

            # CLEAR MEMORY
            del tokenized_answers, tokenized_prompts, tokenized_answers_input_ids, sampling_params, output_answers, labels, log_probs_answers, mask
            torch.cuda.empty_cache()

            return log_probs
               
               
    def batch_prompt(self, 
        prompts: List[str], 
        max_new_tokens: Optional[int] = 500,
        do_sample: Optional[bool] = True,
        top_p: Optional[float] = 0.9,
        temperature: Optional[float] = 0.1,
        num_return_sequences: Optional[int] = 1,
    ) -> List[str]:
        """Text Generation."""
                
        # ENCODE BATCH  
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            n=num_return_sequences,
        )
        
        # SAMPLE NUM_RETURN_SEQUENCES FOR EACH BATCH
        outputs = self.model.generate(
            prompts=prompts,
            sampling_params=sampling_params,
        )
        
        # EXTRACTING GENERATIONS
        generations = []
        for output in outputs: 
            for generated_sequence in output.outputs:
                generations.append(generated_sequence.text)
                
        return generations