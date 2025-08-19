"""
Query decomposer for extracting entity dimensions
"""

import os
from typing import Dict, List, Optional
from collections import defaultdict


class QueryDecomposer:
    """Decompose queries into structured entity dimensions"""
    
    def __init__(self, dimensions: Optional[List[str]] = None, use_gpt: bool = True):
        """
        Initialize query decomposer
        
        Args:
            dimensions: List of entity dimensions
            use_gpt: Whether to use GPT for decomposition
        """
        self.dimensions = dimensions or [
            'location', 'person', 'event', 
            'organization', 'theme', 'date'
        ]
        self.use_gpt = use_gpt
        
        if self.use_gpt:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            except ImportError:
                raise ImportError("Please install openai: pip install openai")
    
    def decompose(self, query: str) -> Dict[str, List[str]]:
        """
        Decompose query into entity dimensions
        
        Args:
            query: Query string
            
        Returns:
            Dictionary mapping dimensions to extracted entities
        """
        if self.use_gpt:
            return self._decompose_with_gpt(query)
        else:
            return self._decompose_simple(query)
    
    def _decompose_with_gpt(self, query: str) -> Dict[str, List[str]]:
        """Use GPT-4o for structured query decomposition"""
        try:
            from pydantic import BaseModel, Field
            from typing import Literal
        except ImportError:
            print("Warning: Pydantic not available, falling back to simple decomposition")
            return self._decompose_simple(query)
        
        # Define structured output schema
        class QueryEntity(BaseModel):
            query_content: str = Field(
                ...,
                description='Entity or phrase to query the documents'
            )
            query_dimension: Literal['person', 'theme', 'event', 'location', 'organization', 'date'] = Field(
                ...,
                description='Dimension of the entity or phrase'
            )
        
        class AllQueries(BaseModel):
            list_of_queries: List[QueryEntity] = Field(
                ...,
                description='List of queries following the required format'
            )
        
        # Build system prompt
        system_prompt = (
            f"You are an expert on question understanding. "
            f"Your task is to:\\n"
            f"1. **Comprehend the given question**: understand what the question asks.\\n"
            f"2. **Extract entities**: identify entities or phrases that belong to these dimensions: {self.dimensions}.\\n"
            f"For each dimension, extract relevant entities that would help retrieve documents to answer the question.\\n\\n"
            f"Example Input:\\n"
            f"Question: How do Indian Monsoons influence Atlantic Hurricane paths?\\n"
            f"Example Output:\\n"
            f"Query 1: dimension='location', content='Atlantic'\\n"
            f"Query 2: dimension='theme', content='Indian Monsoons'\\n"
            f"Query 3: dimension='theme', content='Hurricane paths'"
        )
        
        try:
            # Use structured output with GPT-4o
            response = self.client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': f"Question: {query}"}
                ],
                max_tokens=512,
                temperature=0,
                response_format=AllQueries,
            )
            
            detected_ents = response.choices[0].message.parsed
            
            if detected_ents is None or len(detected_ents.list_of_queries) == 0:
                return {dim: [] for dim in self.dimensions}
            
            # Organize entities by dimension
            cells = defaultdict(list)
            for ent in detected_ents.list_of_queries:
                cells[ent.query_dimension].append(ent.query_content)
            
            return dict(cells)
            
        except Exception as e:
            print(f"GPT decomposition failed: {e}")
            return self._decompose_simple(query)
    
    def _decompose_simple(self, query: str) -> Dict[str, List[str]]:
        """Simple rule-based decomposition as fallback"""
        # This is a very basic implementation
        # In practice, you might want to use NER or other NLP techniques
        cells = {dim: [] for dim in self.dimensions}
        
        # Very simple heuristics
        query_lower = query.lower()
        
        # Look for location indicators
        location_keywords = ['atlantic', 'pacific', 'indian', 'america', 'europe', 'asia', 'africa']
        for keyword in location_keywords:
            if keyword in query_lower:
                cells['location'].append(keyword.capitalize())
        
        # Look for event indicators
        event_keywords = ['hurricane', 'storm', 'flood', 'drought', 'earthquake', 'tornado']
        for keyword in event_keywords:
            if keyword in query_lower:
                cells['event'].append(keyword.capitalize())
        
        # Look for theme indicators
        if 'climate' in query_lower:
            cells['theme'].append('climate change')
        if 'monsoon' in query_lower:
            cells['theme'].append('monsoon')
        if 'path' in query_lower or 'track' in query_lower:
            cells['theme'].append('tracking')
        
        return cells