"""
Optimized data loader for queries, corpus, and hypercube
"""

import os
import json
import glob
import mmap
from typing import List, Dict, Any, Optional, Iterator
from functools import lru_cache


class DataLoader:
    """Load and manage datasets with optimized I/O"""
    
    def __init__(self, dataset_name: str, data_dir: str = "data", cache_data: bool = True):
        """
        Initialize data loader
        
        Args:
            dataset_name: Name of the dataset
            data_dir: Root directory containing data
            cache_data: Whether to cache loaded data in memory
        """
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.cache_data = cache_data
        
        # Cached data
        self._queries_cache = None
        self._corpus_cache = None
        self._hypercube_cache = None
        
    @lru_cache(maxsize=128)
    def _get_file_path(self, data_type: str) -> str:
        """Get file path with caching"""
        return os.path.join(self.data_dir, data_type, f"{self.dataset_name}.jsonl")
    
    def _load_jsonl_streaming(self, file_path: str) -> Iterator[Dict[str, Any]]:
        """Stream JSONL file without loading all into memory"""
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    yield json.loads(line.strip())
    
    def _load_jsonl_fast(self, file_path: str) -> List[Dict[str, Any]]:
        """Fast JSONL loading using memory mapping for large files"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check file size
        file_size = os.path.getsize(file_path)
        
        # For small files (< 10MB), use normal loading
        if file_size < 10 * 1024 * 1024:
            return list(self._load_jsonl_streaming(file_path))
        
        # For large files, use memory mapping
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                for line in mmapped_file.read().decode('utf-8').splitlines():
                    if line.strip():
                        data.append(json.loads(line.strip()))
        
        return data
    
    def load_queries(self, use_cache: bool = None) -> List[Dict[str, Any]]:
        """
        Load query dataset with caching
        
        Args:
            use_cache: Whether to use cached data (overrides instance setting)
            
        Returns:
            List of query dictionaries
        """
        use_cache = use_cache if use_cache is not None else self.cache_data
        
        if use_cache and self._queries_cache is not None:
            return self._queries_cache
        
        query_path = self._get_file_path("query")
        
        queries = self._load_jsonl_fast(query_path)
        
        if use_cache:
            self._queries_cache = queries
        
        print(f"Loaded {len(queries)} queries from {self.dataset_name}")
        return queries
    
    def load_corpus(self, use_cache: bool = None) -> List[Dict[str, Any]]:
        """
        Load document corpus with caching
        
        Args:
            use_cache: Whether to use cached data (overrides instance setting)
            
        Returns:
            List of document dictionaries
        """
        use_cache = use_cache if use_cache is not None else self.cache_data
        
        if use_cache and self._corpus_cache is not None:
            return self._corpus_cache
        
        corpus_path = self._get_file_path("corpus")
        
        corpus = self._load_jsonl_fast(corpus_path)
        
        if use_cache:
            self._corpus_cache = corpus
        
        print(f"Loaded {len(corpus)} documents from {self.dataset_name}")
        return corpus
    
    def stream_queries(self) -> Iterator[Dict[str, Any]]:
        """Stream queries without loading all into memory"""
        query_path = self._get_file_path("query")
        yield from self._load_jsonl_streaming(query_path)
    
    def stream_corpus(self) -> Iterator[Dict[str, Any]]:
        """Stream corpus without loading all into memory"""
        corpus_path = self._get_file_path("corpus")
        yield from self._load_jsonl_streaming(corpus_path)
    
    def get_corpus_path(self) -> str:
        """Get path to corpus file"""
        return self._get_file_path("corpus")
    
    def get_query_count(self) -> int:
        """Get number of queries without loading all data"""
        query_path = self._get_file_path("query")
        count = 0
        with open(query_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    count += 1
        return count
    
    def get_corpus_count(self) -> int:
        """Get number of documents without loading all data"""
        corpus_path = self._get_file_path("corpus")
        count = 0
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    count += 1
        return count
    
    def load_hypercube(self, hypercube_file: Optional[str] = None) -> Optional[Dict[str, Dict]]:
        """
        Load hypercube index if available
        
        Args:
            hypercube_file: Specific hypercube file to use (overrides latest search)
        
        Returns:
            Dictionary mapping doc_id to dimensions, or None if not available
        """
        # Try exact match first
        hypercube_dir = os.path.join(self.data_dir, "hypercube", self.dataset_name)
        
        # For legalbench datasets, try with _contractnli suffix
        if not os.path.exists(hypercube_dir) and self.dataset_name.startswith("legalbench"):
            hypercube_dir = os.path.join(self.data_dir, "hypercube", "legalbench_contractnli")
        
        if not os.path.exists(hypercube_dir):
            print(f"No hypercube index found for {self.dataset_name}")
            return None
        
        # Must specify hypercube file explicitly
        if not hypercube_file:
            # List available files for user reference
            hypercube_files = glob.glob(os.path.join(hypercube_dir, "hypercube_*.jsonl"))
            if hypercube_files:
                file_list = "\n".join([f"  - {os.path.basename(f)}" for f in sorted(hypercube_files)])
                print(f"Error: hypercube_file must be specified. Available files in {hypercube_dir}:\n{file_list}")
            else:
                print(f"No hypercube files found in {hypercube_dir}")
            return None
        
        # Use specified file
        if os.path.isabs(hypercube_file):
            target_file = hypercube_file
        else:
            target_file = os.path.join(hypercube_dir, hypercube_file)
        
        if not os.path.exists(target_file):
            print(f"Specified hypercube file not found: {target_file}")
            return None
        
        print(f"Loading hypercube from: {target_file}")
        
        # Load hypercube as a dictionary for easy lookup
        hypercube = {}
        with open(target_file, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line.strip())
                hypercube[doc['doc_id']] = doc.get('dimensions', {})
        
        self.hypercube = hypercube
        print(f"Loaded hypercube index with {len(hypercube)} documents from {target_file}")
        return hypercube
    
    def get_corpus_path(self) -> str:
        """Get the path to corpus file"""
        return os.path.join(self.data_dir, "corpus", f"{self.dataset_name}.jsonl")
    
    def get_hypercube_path(self) -> Optional[str]:
        """Get the path to hypercube directory"""
        hypercube_dir = os.path.join(self.data_dir, "hypercube", self.dataset_name)
        
        if not os.path.exists(hypercube_dir) and self.dataset_name.startswith("legalbench"):
            hypercube_dir = os.path.join(self.data_dir, "hypercube", "legalbench_contractnli")
        
        if os.path.exists(hypercube_dir):
            return hypercube_dir
        return None