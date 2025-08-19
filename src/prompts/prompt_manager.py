"""
Prompt manager for different datasets and retrieval methods
"""

from typing import List, Optional


class PromptManager:
    """Manage prompts for different datasets and configurations"""
    
    # Dataset-specific RAG prompts
    RAG_PROMPTS = {
        # Scientific datasets with quantitative focus
        'hurricane': (
            'Answer the query based on the given retrieved documents. '
            'If the query asks the quantitative analysis, such as starting with "How many", "How much", '
            '"How much greater", "How wide", "What percentage", "How far", "How long", "How old", '
            '"What portion", "What depth", you must directly output the quantitative answer as short as possible. '
            'If the query starting with "what percentage" or the query includes the word "likelihood", '
            'please directly output the number with %. '
            'If the query starting with "what specific", "which years", "who", '
            'please directly output the answer without explanation or other information. '
            'Otherwise, please provide as precise information as possible based the retrieved documents.'
        ),
        
        'geography': (
            'Answer the query based on the given retrieved documents. '
            'If the query asks the quantitative analysis, such as starting with "how many", "how much", '
            '"how much greater", "how wide", "what percentage", "how far", "how long", "how old", '
            '"what portion", "what depth", you must output the quantitative answer as short as possible. '
            'If the query starting with "what percentage" or the query includes the word "likelihood", '
            'please directly output the number with %. '
            'If the query starting with "what specific", "which years", "who", '
            'please directly output the answer without explanation or other information. '
            'Otherwise, please provide as precise information as possible based the retrieved documents.'
        ),
        
        'aging_dam': (
            'Answer the query based on the given retrieved documents. '
            'If the query asks the quantitative analysis, such as starting with "How many", "How much", '
            '"How much greater", "How wide", "What percentage", "How far", "How long", "How old", '
            '"What portion", "What depth", you must directly output the quantitative answer as short as possible. '
            'If the query starting with "what percentage" or the query includes the word "likelihood", '
            'please directly output the number with %. '
            'If the query starting with "what specific", "which years", "who", '
            'please directly output the answer without explanation or other information. '
            'Otherwise, please provide as precise information as possible based the retrieved documents.'
        ),
        
        # Legal datasets with precision focus
        'legalbench': (
            'Answer the query based on the given retrieved documents. '
            'Please keep the answer as precise as possible.'
        ),
        
        'legalbench_contractnli': (
            'Answer the query based on the given retrieved documents. '
            'Please keep the answer as precise as possible.'
        ),
        
        'legalbench_cuad': (
            'Answer the query based on the given retrieved documents. '
            'Please keep the answer as precise as possible.'
        ),
        
        'legalbench_maud': (
            'Answer the query based on the given retrieved documents. '
            'Please keep the answer as precise as possible.'
        ),
        
        'legalbench_privacy_qa': (
            'Answer the query based on the given retrieved documents. '
            'Please keep the answer as precise as possible.'
        ),
        
        # Scientific fact verification
        'scifact': (
            'Answer the query based on the given retrieved documents. '
            'If the query asks the quantitative analysis, such as starting with "How many", "How much", '
            '"How much greater", "How wide", "What percentage", "How far", "How long", "How old", '
            '"What portion", "What depth", you must directly output the quantitative answer as short as possible. '
            'If the query starting with "what percentage" or the query includes the word "likelihood", '
            'please directly output the number with %. '
            'If the query starting with "what specific", "which years", "who", '
            'please directly output the answer without explanation or other information. '
            'Otherwise, please provide as precise information as possible based the retrieved documents.'
        ),
        
        'hydrology': (
            'Answer the query based on the given retrieved documents. '
            'If the query asks the quantitative analysis, such as starting with "How many", "How much", '
            '"How much greater", "How wide", "What percentage", "How far", "How long", "How old", '
            '"What portion", "What depth", you must directly output the quantitative answer as short as possible. '
            'If the query starting with "what percentage" or the query includes the word "likelihood", '
            'please directly output the number with %. '
            'If the query starting with "what specific", "which years", "who", '
            'please directly output the answer without explanation or other information. '
            'Otherwise, please provide as precise information as possible based the retrieved documents.'
        ),
        
        # Default prompt
        'default': (
            'Answer the query based on the given retrieved documents. '
            'Please provide as precise information as possible based the retrieved documents.'
        )
    }
    
    # No-RAG prompt (direct inference)
    NO_RAG_PROMPT = (
        "You are a helpful question answering assistant. "
        "Please output the answers as short as possible without punctuation."
    )
    
    @classmethod
    def get_rag_prompt(cls, dataset: str, query: str, documents: List[str]) -> str:
        """
        Get RAG prompt for a specific dataset
        
        Args:
            dataset: Dataset name
            query: User query
            documents: List of retrieved document contents
            
        Returns:
            Formatted prompt string
        """
        # Get dataset-specific instruction
        instruction = cls.RAG_PROMPTS.get(dataset, cls.RAG_PROMPTS['default'])
        
        # Format documents
        doc_text = '\n\n'.join([
            f"Document {i+1}: {doc}" 
            for i, doc in enumerate(documents)
        ])
        
        # Build complete prompt
        return f"{instruction}\n\nDocuments:\n{doc_text}\n\nQuery: {query}\nAnswer:"
    
    @classmethod
    def get_no_rag_prompt(cls, query: str) -> str:
        """
        Get no-RAG prompt for direct inference
        
        Args:
            query: User query
            
        Returns:
            Formatted prompt string
        """
        return f"{cls.NO_RAG_PROMPT}\n\nQuery: {query}\nAnswer:"
    
    @classmethod
    def get_system_prompt(cls, dataset: str, use_rag: bool = True) -> str:
        """
        Get system prompt for a specific configuration
        
        Args:
            dataset: Dataset name
            use_rag: Whether using RAG or direct inference
            
        Returns:
            System prompt string
        """
        if use_rag:
            return cls.RAG_PROMPTS.get(dataset, cls.RAG_PROMPTS['default'])
        else:
            return cls.NO_RAG_PROMPT