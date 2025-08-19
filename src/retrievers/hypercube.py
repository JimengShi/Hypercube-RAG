"""
Hypercube retriever implementation
"""

import os
import json
import glob
import pickle
from collections import defaultdict, Counter
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from .base import Retriever


class HypercubeRetriever(Retriever):
    """Hypercube retrieval using multi-dimensional entity indexing"""
    
    def __init__(
        self, 
        corpus_path: str,
        hypercube_path: Optional[str] = None,
        hypercube_file: Optional[str] = None,
        embedding_model: str = 'intfloat/e5-base-v2',
        embedding_cache_path: str = './ent2emb.pkl',
        **kwargs
    ):
        """
        Initialize hypercube retriever
        
        Args:
            corpus_path: Path to corpus JSONL file
            hypercube_path: Path to hypercube directory
            hypercube_file: Specific hypercube file to use (overrides path-based search)
            embedding_model: Model for entity embeddings
            embedding_cache_path: Path to cache entity embeddings
            **kwargs: Additional configuration
        """
        super().__init__(corpus_path, **kwargs)
        self.hypercube_path = hypercube_path
        self.hypercube_file = hypercube_file
        self.embedding_model_name = embedding_model
        self.embedding_cache_path = embedding_cache_path
        
        self.encoder = None
        self.ent2emb = None
        self.hypercube = defaultdict(lambda: defaultdict(list))
        self.dimensions = ['location', 'person', 'event', 'organization', 'theme', 'date']
        
        self.build_index()
    
    def build_index(self):
        """Build hypercube index from files"""
        print("Building hypercube index...")
        
        # Load encoder for entity embeddings
        self.encoder = SentenceTransformer(self.embedding_model_name)
        
        # Load entity embedding cache
        self._load_embedding_cache()
        
        # Find hypercube path if not provided
        if self.hypercube_path is None:
            self.hypercube_path = self._find_hypercube_path()
        
        if self.hypercube_path is None:
            raise ValueError(f"No hypercube index found for corpus")
        
        # Load hypercube data
        self._load_hypercube()
        
        print(f"Hypercube index built with {len(self.dimensions)} dimensions")
    
    def _find_hypercube_path(self) -> Optional[str]:
        """Find hypercube path based on corpus path"""
        # Extract dataset name from corpus path
        corpus_name = os.path.basename(self.corpus_path).replace('.jsonl', '')
        
        # Look for hypercube directory
        data_dir = os.path.dirname(os.path.dirname(self.corpus_path))
        hypercube_dir = os.path.join(data_dir, 'hypercube', corpus_name)
        
        # Special case for legalbench
        if not os.path.exists(hypercube_dir) and corpus_name.startswith('legalbench'):
            hypercube_dir = os.path.join(data_dir, 'hypercube', 'legalbench_contractnli')
        
        if os.path.exists(hypercube_dir):
            return hypercube_dir
        
        return None
    
    def _load_hypercube(self):
        """Load hypercube from JSONL file"""
        if not self.hypercube_file:
            # List available files for user reference
            hypercube_files = glob.glob(os.path.join(self.hypercube_path, 'hypercube_*.jsonl'))
            if hypercube_files:
                file_list = "\n".join([f"  - {os.path.basename(f)}" for f in sorted(hypercube_files)])
                raise ValueError(f"hypercube_file must be specified. Available files in {self.hypercube_path}:\n{file_list}")
            else:
                raise FileNotFoundError(f"No hypercube files found in {self.hypercube_path}")
        
        # Use specified file
        if os.path.isabs(self.hypercube_file):
            hypercube_file = self.hypercube_file
        else:
            hypercube_file = os.path.join(self.hypercube_path, self.hypercube_file)
        
        if not os.path.exists(hypercube_file):
            raise FileNotFoundError(f"Specified hypercube file not found: {hypercube_file}")
        
        print(f"Loading hypercube from: {hypercube_file}")
        
        # Collect all unique entities first
        new_entities = set()
        
        # Build inverted index: entity -> doc_ids
        with open(hypercube_file, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line.strip())
                doc_id = doc['doc_id']
                
                # Find the document index in corpus
                if doc_id in self.doc_id_to_idx:
                    doc_idx = self.doc_id_to_idx[doc_id]
                    
                    # Add to inverted index
                    for dimension, entities in doc.get('dimensions', {}).items():
                        for entity, count in entities.items():
                            self.hypercube[dimension][entity].append(doc_idx)
                            # Collect new entities for batch embedding
                            if entity not in self.ent2emb:
                                new_entities.add(entity)
        
        # Batch compute embeddings for new entities
        if new_entities:
            print(f"Computing embeddings for {len(new_entities)} new entities...")
            for i, entity in enumerate(new_entities):
                if i % 100 == 0:
                    print(f"  Processing entity {i}/{len(new_entities)}...")
                self._embed_entity_no_save(entity)
            # Save all embeddings once at the end
            self._save_embedding_cache()
            print(f"Saved {len(self.ent2emb)} total entity embeddings to cache")
    
    def _load_embedding_cache(self):
        """Load cached entity embeddings"""
        if os.path.exists(self.embedding_cache_path):
            with open(self.embedding_cache_path, 'rb') as f:
                self.ent2emb = pickle.load(f)
                print(f"Loaded {len(self.ent2emb)} cached entity embeddings")
        else:
            self.ent2emb = {}
            print("Initialized new entity embedding cache")
    
    def _save_embedding_cache(self):
        """Save entity embeddings to cache"""
        with open(self.embedding_cache_path, 'wb') as f:
            pickle.dump(self.ent2emb, f)
    
    def _embed_entity(self, entity: str) -> np.ndarray:
        """Get or compute embedding for an entity (with cache save)"""
        if entity not in self.ent2emb:
            # Use query prefix for E5 models
            if 'e5' in self.embedding_model_name.lower():
                text = 'query: ' + entity
            else:
                text = entity
            
            self.ent2emb[entity] = self.encoder.encode(text, normalize_embeddings=True)
            self._save_embedding_cache()
        
        return self.ent2emb[entity]
    
    def _embed_entity_no_save(self, entity: str) -> np.ndarray:
        """Get or compute embedding for an entity (without cache save)"""
        if entity not in self.ent2emb:
            # Use query prefix for E5 models
            if 'e5' in self.embedding_model_name.lower():
                text = 'query: ' + entity
            else:
                text = entity
            
            self.ent2emb[entity] = self.encoder.encode(text, normalize_embeddings=True)
        
        return self.ent2emb[entity]
    
    def _find_similar_entities(self, query_entity: str, dimension: str, threshold: float = 0.9) -> List[str]:
        """Find similar entities in a dimension using embedding similarity"""
        query_emb = self._embed_entity(query_entity)
        similar_entities = []
        
        for entity in self.hypercube[dimension].keys():
            entity_emb = self._embed_entity(entity)
            similarity = np.dot(query_emb, entity_emb)
            
            if similarity >= threshold:
                similar_entities.append(entity)
        
        return similar_entities
    
    def retrieve_by_entities(self, cells: Dict[str, List[str]], k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve documents using entity cells
        
        Args:
            cells: Dictionary mapping dimensions to entity lists
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        # Collect all document indices
        doc_indices = []
        
        for dimension, entities in cells.items():
            if dimension not in self.dimensions:
                continue
            
            for entity in entities:
                # Try exact match first
                if entity in self.hypercube[dimension]:
                    doc_indices.extend(self.hypercube[dimension][entity])
                else:
                    # Try similarity-based matching
                    similar_entities = self._find_similar_entities(entity, dimension)
                    for similar_entity in similar_entities:
                        doc_indices.extend(self.hypercube[dimension][similar_entity])
        
        # Get top-k most frequent documents
        doc_counts = Counter(doc_indices)
        top_docs = doc_counts.most_common(k)
        
        # Build result list
        results = []
        for doc_idx, count in top_docs:
            results.append({
                'doc_id': self.corpus[doc_idx]['doc_id'],
                'content': self.corpus[doc_idx]['content'],
                'score': float(count),  # Use frequency as score
                'rank': len(results) + 1
            })
        
        return results
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top-k documents using hypercube
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        # Use query decomposer to extract entities
        from ..query import QueryDecomposer
        decomposer = QueryDecomposer(dimensions=self.dimensions)
        cells = decomposer.decompose(query)
        
        print(f"Decomposed query into cells: {cells}")
        
        # Retrieve using extracted entities
        return self.retrieve_by_entities(cells, k)