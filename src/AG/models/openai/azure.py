from typing import List, Dict

import os

from openai import AsyncAzureOpenAI


class AsyncAzureChatLLM:
    """
    Wrapper for an (Async) Azure Chat Model.
    """
    def __init__(
        self, 
        azure_endpoint: str, 
        api_version: str, 
        ):
        """
        Initializes AsyncAzureOpenAI client.
        """
        self.client = AsyncAzureOpenAI(
            api_version=api_version,
            api_key=os.getenv("OPENAI_API_KEY"),
            azure_endpoint=azure_endpoint,
        )

    @property
    def llm_type(self):
        return "AsyncAzureOpenAI"

    async def __call__(self, 
        messages: List[Dict[str, str]], 
        **kwargs,
    ):
        """
        Make an async API call.
        """
        return await self.client.chat.completions.create(
            messages=messages, 
            **kwargs)