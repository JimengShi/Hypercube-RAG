"""
Base retriever class
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import json


class Retriever(ABC):
    """Abstract base class for all retrieval methods"""
    
    def __init__(self, corpus_path: str, **kwargs):
        """
        Initialize retriever with corpus
        
        Args:
            corpus_path: Path to corpus file (JSONL format)
            **kwargs: Additional configuration parameters
        """
        self.corpus_path = corpus_path
        self.corpus = []
        self.doc_id_to_idx = {}
        self.config = kwargs
        self._load_corpus()
    
    def _load_corpus(self):
        """Load corpus from JSONL file"""
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                doc = json.loads(line.strip())
                self.corpus.append(doc)
                self.doc_id_to_idx[doc['doc_id']] = idx
        print(f"Loaded {len(self.corpus)} documents from {self.corpus_path}")
    
    @abstractmethod
    def build_index(self):
        """Build retrieval index from corpus"""
        pass
    
    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top-k documents for a query
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents with metadata
        """
        pass