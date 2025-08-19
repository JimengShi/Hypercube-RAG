"""
Dense retriever implementation using sentence transformers
"""

import numpy as np
import pickle
import os
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from .base import Retriever


class DenseRetriever(Retriever):
    """Dense retrieval using sentence embeddings"""
    
    def __init__(self, corpus_path: str, model_name: str = 'intfloat/e5-base-v2', 
                 cache_embeddings: bool = True, batch_size: int = 32, **kwargs):
        """
        Initialize dense retriever
        
        Args:
            corpus_path: Path to corpus JSONL file
            model_name: Sentence transformer model name
            cache_embeddings: Whether to cache document embeddings
            batch_size: Batch size for encoding
            **kwargs: Additional configuration
        """
        super().__init__(corpus_path, **kwargs)
        self.model_name = model_name
        self.cache_embeddings = cache_embeddings
        self.batch_size = batch_size
        self.encoder = None
        self.doc_embeddings = None
        self.is_e5_model = 'e5' in model_name.lower()
        
        # Cache path for embeddings
        corpus_name = os.path.splitext(os.path.basename(corpus_path))[0]
        model_safe_name = model_name.replace('/', '_')
        self.cache_path = f"embeddings_{corpus_name}_{model_safe_name}.pkl"
        
        self.build_index()
    
    def build_index(self):
        """Build embedding index from corpus"""
        # Try to load from cache first
        if self.cache_embeddings and os.path.exists(self.cache_path):
            print(f"Loading cached embeddings from {self.cache_path}...")
            with open(self.cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                self.doc_embeddings = cache_data['embeddings']
                stored_model = cache_data.get('model_name', '')
                if stored_model != self.model_name:
                    print(f"Warning: Cached embeddings from different model ({stored_model}), rebuilding...")
                    self._build_embeddings()
                else:
                    print(f"Loaded {len(self.doc_embeddings)} cached embeddings")
                    # Still need to load the encoder for query encoding
                    self.encoder = SentenceTransformer(self.model_name)
        else:
            self._build_embeddings()
    
    def _build_embeddings(self):
        """Build embeddings from scratch"""
        print(f"Building dense index with {self.model_name}...")
        
        # Load encoder model
        self.encoder = SentenceTransformer(self.model_name)
        
        # Prepare documents with prefix for E5 models
        documents = []
        for doc in self.corpus:
            text = doc['content']
            if self.is_e5_model:
                text = 'passage: ' + text
            documents.append(text)
        
        # Encode all documents in batches
        print(f"Encoding {len(documents)} documents...")
        self.doc_embeddings = self.encoder.encode(
            documents, 
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=self.batch_size,
            convert_to_numpy=True  # Ensure numpy array for efficiency
        )
        
        # Cache embeddings if enabled
        if self.cache_embeddings:
            print(f"Caching embeddings to {self.cache_path}...")
            with open(self.cache_path, 'wb') as f:
                pickle.dump({
                    'embeddings': self.doc_embeddings,
                    'model_name': self.model_name
                }, f)
        
        print(f"Dense index built with shape {self.doc_embeddings.shape}")
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top-k documents using dense retrieval
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents with scores
        """
        if self.doc_embeddings is None:
            raise RuntimeError("Embeddings not built. Call build_index() first.")
        
        # Encode query with appropriate prefix
        query_text = f'query: {query}' if self.is_e5_model else query
        
        query_embedding = self.encoder.encode(
            query_text,  # Pass as string, not list for single query
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        # Calculate cosine similarity scores (more efficient with @ operator)
        scores = self.doc_embeddings @ query_embedding
        
        # Get top-k indices using argpartition (more efficient than full sort)
        if k < len(scores):
            # argpartition is O(n) while argsort is O(n log n)
            top_k_unsorted = np.argpartition(scores, -k)[-k:]
            top_indices = top_k_unsorted[np.argsort(scores[top_k_unsorted])[::-1]]
        else:
            top_indices = np.argsort(scores)[::-1][:k]
        
        # Build result list
        results = []
        for idx in top_indices:
            results.append({
                'doc_id': self.corpus[idx]['doc_id'],
                'content': self.corpus[idx]['content'],
                'score': float(scores[idx]),
                'rank': len(results) + 1
            })
        
        return results