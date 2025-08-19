"""
Base LLM class
"""

from abc import ABC, abstractmethod
from typing import List, Optional


class LLM(ABC):
    """Abstract base class for Language Model wrappers"""
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize LLM wrapper
        
        Args:
            model_name: Name of the model
            **kwargs: Additional configuration (temperature, max_tokens, etc.)
        """
        self.model_name = model_name
        self.config = kwargs
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate response from LLM
        
        Args:
            prompt: Complete prompt including context
            
        Returns:
            Generated response string
        """
        pass
    
    @abstractmethod
    def batch_generate(self, prompts: List[str]) -> List[str]:
        """
        Generate responses for multiple prompts in batch
        
        Args:
            prompts: List of prompts
            
        Returns:
            List of generated responses
        """
        pass