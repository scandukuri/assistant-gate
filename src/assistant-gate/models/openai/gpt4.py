from typing import (
    Union,
    Any,
    Dict,
    List,
)
import time

import asyncio
import logging
logging.basicConfig(level=logging.INFO)
 # The cost per token for each model input.
MODEL_COST_PER_INPUT = {
    'gpt-4': 3e-05,
}
# The cost per token for each model output.
MODEL_COST_PER_OUTPUT = {
    'gpt-4': 6e-05,
}
class GPT4Agent():
    """
    gpt-4 LLM wrapper for async API calls.
    """
    def __init__(
        self,
        llm: Any,
        **completion_config,
    ) -> None:
        self.llm = llm
        self.completion_config = completion_config
        self.all_responses = []
        self.total_inference_cost = 0
    def calc_cost(
        self,
        response
    ) -> float:
        """
        Args:
        response (openai.ChatCompletion): The response from the API.
        Returns:
        float: The cost of the response.
        """
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cost = (
            MODEL_COST_PER_INPUT['gpt-4'] * input_tokens
            + MODEL_COST_PER_OUTPUT['gpt-4'] * output_tokens
        )
        return cost
    
    
    def get_prompt(
        self,
        system_message: str,
        user_message: Union[str, List[str]],
    ) -> List[Dict[str, str]]:
        """
        Get the (zero- or few- shot) prompt for the (chat) model.
        """
        if type(user_message) == str:    
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ]
        else:
            messages = [{"role": "system", "content": system_message}]
            messages.extend([{"role": "user", "content": message} if i % 2 == 0 else {"role": "assistant", "content": message} for i, message in enumerate(user_message)])
        return messages
    
    
    async def get_response(
        self,
        messages: List[Dict[str, str]],
    ) -> Any:
        """
        Get the response from the model.
        """
        return await self.llm(messages=messages, **self.completion_config)
    
    
    async def run(
        self,
        system_message: str,
        message: str,
    ) -> Dict[str, Any]:
        """Runs the Code Agent
        Args:
            system_message (str): The system message to use
            message (str): The user message to use
        Returns:
            A dictionary containing the code model's response and the cost of the performed API call
        """
        messages = self.get_prompt(system_message=system_message, user_message=message)
        for i in range(100):
            try:
                response = await self.get_response(messages=messages)
                break
            except:
                time.sleep(1)
        #response = await self.get_response(messages=messages)
        cost = self.calc_cost(response=response)
        logging.info(f"Cost for running gpt4: {cost}")
        full_response = {
            'response': response,
            'response_str': [r.message.content for r in response.choices],
            'cost': cost
        }
        # Update total cost and store response
        self.total_inference_cost += cost
        self.all_responses.append(full_response)
        return full_response['response_str']
    
    
    async def batch_prompt_sync(
        self,
        system_message: str,
        messages: List[str],
    ) -> List[str]:
        """Handles async API calls for batch prompting.
        Args:
            system_message (str): The system message to use
            messages (List[str]): A list of user messages
        Returns:
            A list of responses from the code model for each message
        """
        responses = [self.run(system_message, message) for message in messages]
        return await asyncio.gather(*responses)
    
    
    def batch_prompt(
        self,
        system_message: str,
        messages: List[str],
    ) -> List[str]:
        """=
        Synchronous wrapper for batch_prompt.
        Args:
            system_message (str): The system message to use
            messages (List[str]): A list of user messages
        Returns:
            A list of responses from the code model for each message
        """
        loop = asyncio.get_event_loop()
        if loop.is_running():
            raise RuntimeError(f"Loop is already running.")
        return loop.run_until_complete(self.batch_prompt_sync(system_message, messages))