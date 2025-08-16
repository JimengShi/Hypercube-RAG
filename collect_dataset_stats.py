#!/usr/bin/env python3
"""
Collect statistics for Hypercube-RAG dataset
"""

import json
from pathlib import Path

def collect_stats():
    stats = {}
    
    # Collect query stats
    query_dir = Path("data/query")
    corpus_dir = Path("data/corpus")
    
    datasets = [
        "hurricane", "geography", "aging_dam", "scifact", 
        "legalbench_contractnli", "legalbench_cuad", "legalbench_maud", "legalbench_privacy_qa",
        "hydrology"
    ]
    
    total_queries = 0
    total_docs = 0
    
    for dataset in datasets:
        query_file = query_dir / f"{dataset}.jsonl"
        corpus_file = corpus_dir / f"{dataset}.jsonl"
        
        # Count queries
        query_count = 0
        if query_file.exists():
            with open(query_file, 'r') as f:
                query_count = sum(1 for line in f)
        
        # Count documents
        doc_count = 0
        if corpus_file.exists():
            with open(corpus_file, 'r') as f:
                doc_count = sum(1 for line in f)
        
        stats[dataset] = {
            "queries": query_count,
            "documents": doc_count
        }
        
        total_queries += query_count
        total_docs += doc_count
    
    stats["total"] = {
        "queries": total_queries,
        "documents": total_docs,
        "datasets": len(datasets)
    }
    
    return stats

def main():
    stats = collect_stats()
    
    print("Hypercube-RAG Dataset Statistics")
    print("=" * 40)
    
    for dataset, data in stats.items():
        if dataset == "total":
            print(f"\nTOTAL:")
            print(f"  Datasets: {data['datasets']}")
            print(f"  Queries: {data['queries']:,}")
            print(f"  Documents: {data['documents']:,}")
        else:
            print(f"{dataset}:")
            print(f"  Queries: {data['queries']:,}")
            print(f"  Documents: {data['documents']:,}")
    
    # Save to JSON
    with open("dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nStatistics saved to dataset_stats.json")

if __name__ == "__main__":
    main()