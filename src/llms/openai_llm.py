"""
OpenAI LLM wrapper
"""

import os
from typing import List, Optional
from .base import LLM


class OpenAILLM(LLM):
    """Wrapper for OpenAI models (GPT-4, GPT-4o, etc.)"""
    
    # Model name mappings
    MODEL_MAPPINGS = {
        "gpt-4o": "gpt-4o-2024-11-20",
        "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
        "gpt-4.1": "gpt-4.1-2025-04-14",
        "gpt-4.1-mini": "gpt-4.1-mini-2025-04-14",
        "gpt-4.1-nano": "gpt-4.1-nano-2025-04-14"
    }
    
    def __init__(self, model_name: str = "gpt-4o", **kwargs):
        """
        Initialize OpenAI LLM
        
        Args:
            model_name: Model name (gpt-4, gpt-4o, gpt-3.5-turbo, etc.)
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
        """
        # Map friendly name to actual model name
        actual_model_name = self.MODEL_MAPPINGS.get(model_name, model_name)
        super().__init__(actual_model_name, **kwargs)
        
        # Store both names for reference
        self.friendly_name = model_name
        self.actual_model_name = actual_model_name
        
        # Lazy import to avoid dependency issues
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
    
    def generate(self, prompt: str) -> str:
        """
        Generate response using OpenAI API
        
        Args:
            prompt: Complete prompt
            
        Returns:
            Generated response
        """
        # Default parameters
        params = {
            'temperature': 0.7,
            'max_tokens': 512,
            **self.config
        }
        
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            **params
        )
        
        return completion.choices[0].message.content
    
    def batch_generate(self, prompts: List[str]) -> List[str]:
        """
        Generate responses for multiple prompts
        
        Args:
            prompts: List of prompts
            
        Returns:
            List of responses
        """
        # For now, use sequential generation
        # TODO: Implement parallel processing using the parallel API from legacy
        responses = []
        for prompt in prompts:
            responses.append(self.generate(prompt))
        return responses