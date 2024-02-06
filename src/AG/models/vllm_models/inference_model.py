import os
import logging 
from typing import Optional, List, Tuple

import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)


class VLLMInferenceModel():
    """Wrapper for running inference with VLLM."""
    def __init__(
        self, 
        model: str,
        download_dir: str,
        dtype: str,
        tensor_parallel_size: int,
        seed: int = 1,
    ):
        """Initializes VLLM Inference Model"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model,
            cache_dir=download_dir,
            token=os.getenv("HF_TOKEN"),
        )
        
        self.tokenizer.pad_token = "[PAD]"
        self.tokenizer.padding_side = "right"

        self.model = LLM(
            model=model,
            download_dir=download_dir,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
        )
        
    @property
    def model_type(self):
        return "VLLMInferenceModel"
    
    def batch_log_probs(
        self, 
        prompts: List[str],
        answers: List[str], 
    ) -> torch.Tensor:
        """Returns log probabilities of prompts including answer (answers) and prompts excluding answers (prompts)."""
        with torch.no_grad():
            # tokenize answers first to get max length
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
                use_tqdm=False,
            )
  
            # now get the tokens back
            log_probs_answers = torch.tensor([
                [v for prob in output_answer.prompt_logprobs[1:] for _, v in prob.items()]
                for output_answer in output_answers
            ])
            
            
            # mask answers 
            mask_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id != 0 else 0
            labels = tokenized_answers.input_ids[:, 1:]
            mask = torch.logical_and(tokenized_prompts.input_ids[:, 1:] == mask_id, labels != 0)
            log_probs_answers.masked_fill_(~mask, 0) 
            log_probs = log_probs_answers.sum(dim=-1)

            # clear memory (prob not needed)
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
        """Batched text generation."""       
        # sampling params
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            n=num_return_sequences,
        )
        
        # sample
        outputs = self.model.generate(
            prompts=prompts,
            sampling_params=sampling_params,
        )
        
        # extract generations
        generations = []
        for output in outputs: 
            for generated_sequence in output.outputs:
                generations.append(generated_sequence.text)
                
        return generations