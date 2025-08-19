"""
BM25 retriever implementation
"""

import numpy as np
import pickle
import os
import re
from typing import List, Dict, Any, Set
from rank_bm25 import BM25Okapi
from .base import Retriever


class BM25Retriever(Retriever):
    """BM25 retrieval using rank_bm25"""
    
    def __init__(self, corpus_path: str, use_cache: bool = True, **kwargs):
        """
        Initialize BM25 retriever
        
        Args:
            corpus_path: Path to corpus JSONL file
            use_cache: Whether to cache the BM25 index
            **kwargs: Additional configuration
        """
        super().__init__(corpus_path, **kwargs)
        self.use_cache = use_cache
        self.bm25 = None
        self.tokenized_corpus = []
        
        # Cache path for BM25 index
        corpus_name = os.path.splitext(os.path.basename(corpus_path))[0]
        self.cache_path = f"bm25_index_{corpus_name}.pkl"
        
        # Precompile regex for tokenization
        self.token_pattern = re.compile(r'\b\w+\b')
        
        self.build_index()
    
    def _tokenize(self, text: str) -> List[str]:
        """Improved tokenization with regex (faster than split for complex text)"""
        return self.token_pattern.findall(text.lower())
    
    def build_index(self):
        """Build BM25 index from corpus"""
        # Try to load from cache first
        if self.use_cache and os.path.exists(self.cache_path):
            print(f"Loading cached BM25 index from {self.cache_path}...")
            try:
                with open(self.cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.bm25 = cache_data['bm25']
                    self.tokenized_corpus = cache_data['tokenized_corpus']
                    print(f"Loaded BM25 index with {len(self.tokenized_corpus)} documents")
                    return
            except Exception as e:
                print(f"Failed to load cache: {e}, rebuilding...")
        
        print("Building BM25 index...")
        
        # Tokenize corpus with improved tokenization
        self.tokenized_corpus = [
            self._tokenize(doc['content'])
            for doc in self.corpus
        ]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        # Cache the index if enabled
        if self.use_cache:
            print(f"Caching BM25 index to {self.cache_path}...")
            try:
                with open(self.cache_path, 'wb') as f:
                    pickle.dump({
                        'bm25': self.bm25,
                        'tokenized_corpus': self.tokenized_corpus
                    }, f)
            except Exception as e:
                print(f"Failed to cache index: {e}")
        
        print(f"BM25 index built with {len(self.tokenized_corpus)} documents")
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top-k documents using BM25
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents with scores
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built. Call build_index() first.")
        
        # Tokenize query using improved tokenization
        tokenized_query = self._tokenize(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k document indices using argpartition for efficiency
        if k < len(scores):
            # argpartition is O(n) while argsort is O(n log n)
            top_k_unsorted = np.argpartition(scores, -k)[-k:]
            top_indices = top_k_unsorted[np.argsort(scores[top_k_unsorted])[::-1]]
        else:
            top_indices = np.argsort(scores)[::-1][:k]
        
        # Build result list - filter out zero scores
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include documents with positive scores
                results.append({
                    'doc_id': self.corpus[idx]['doc_id'],
                    'content': self.corpus[idx]['content'],
                    'score': float(scores[idx]),
                    'rank': len(results) + 1
                })
        
        return results