"""
Language Model wrappers
"""

from .base import LLM
from .openai_llm import OpenAILLM

__all__ = [
    'LLM',
    'OpenAILLM'
]