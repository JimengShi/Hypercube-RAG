#!/usr/bin/env python3
"""
Unified pipeline for Hypercube-RAG experiments
"""

import os
import json
import argparse
import time
from typing import Dict, List, Any, Optional
from tqdm import tqdm

# Import our modules
from src.data import DataLoader
from src.retrievers import BM25Retriever, DenseRetriever, HypercubeRetriever
from src.llms import OpenAILLM
from src.prompts import PromptManager
from src.evaluation import NLPMetrics, LLMJudge
from src.config import ConfigManager


def _build_parser():
    """Build argument parser"""
    parser = argparse.ArgumentParser(description="Hypercube-RAG Unified Pipeline")
    
    # Config file argument
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset", 
        type=str,
        help="Dataset name (e.g., hurricane, geography, legalbench_contractnli)"
    )
    
    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        choices=["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"],
        help="LLM model to use"
    )
    
    # Retrieval arguments
    parser.add_argument(
        "--retrieval_method",
        type=str,
        default="hypercube",
        choices=["none", "hypercube", "dense", "bm25"],
        help="Retrieval method to use"
    )
    
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="intfloat/e5-base-v2",
        help="Embedding model for dense retrieval"
    )
    
    parser.add_argument(
        "--hypercube_file",
        type=str,
        default=None,
        help="Specific hypercube file to use (if not specified, uses latest)"
    )
    
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of documents to retrieve"
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--eval_metrics",
        type=str,
        nargs="+",
        default=["exact_match", "f1", "rouge"],
        help="Evaluation metrics to compute"
    )
    
    parser.add_argument(
        "--use_llm_judge",
        action="store_true",
        help="Use LLM as judge for evaluation"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save results"
    )
    
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to file"
    )
    
    # Other arguments
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for debugging)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress"
    )
    
    return parser


def initialize_retriever(args, corpus_path: str) -> Optional[Any]:
    """Initialize the retriever based on arguments"""
    if args.retrieval_method == "none":
        return None
    elif args.retrieval_method == "bm25":
        return BM25Retriever(corpus_path)
    elif args.retrieval_method == "dense":
        return DenseRetriever(corpus_path, model_name=args.embedding_model)
    elif args.retrieval_method == "hypercube":
        return HypercubeRetriever(
            corpus_path, 
            embedding_model=args.embedding_model,
            hypercube_file=args.hypercube_file
        )
    else:
        raise ValueError(f"Unknown retrieval method: {args.retrieval_method}")


def initialize_llm(args) -> Any:
    """Initialize the LLM based on arguments"""
    if args.model in ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"]:
        return OpenAILLM(model_name=args.model)
    else:
        raise ValueError(f"Unknown model: {args.model}")


def run_pipeline(args):
    """Run the main pipeline"""
    print("=" * 80)
    print("Hypercube-RAG")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Retrieval: {args.retrieval_method}")
    print(f"Top-K: {args.k}")
    print(f"Embedding Model: {args.embedding_model}")
    print(f"Max Samples: {args.max_samples}")
    print(f"Save Results: {args.save}")
    print(f"Verbose: {args.verbose}")
    print("=" * 80)
    
    # Load data
    print("\n📚 Loading data...")
    data_loader = DataLoader(args.dataset)
    queries = data_loader.load_queries()
    corpus_path = data_loader.get_corpus_path()
    
    # Limit samples if specified
    if args.max_samples:
        queries = queries[:args.max_samples]
        print(f"Limited to {len(queries)} samples")
    
    # Initialize retriever
    retriever = None
    if args.retrieval_method != "none":
        print(f"\n🔍 Initializing {args.retrieval_method} retriever...")
        retriever = initialize_retriever(args, corpus_path)
    
    # Initialize LLM
    print(f"\n🤖 Initializing {args.model} LLM...")
    llm = initialize_llm(args)
    
    # Process queries
    print(f"\n🚀 Processing {len(queries)} queries...")
    results = []
    predictions = []
    references = []
    
    for i, query_item in enumerate(tqdm(queries, desc="Processing")):
        query = query_item['content']
        reference = query_item.get('answer_list', [''])[0]  # Take first answer as reference
        
        if args.verbose:
            print(f"\nQ{i+1}: {query}")
        
        # Retrieve documents if using RAG
        retrieved_docs = []
        if retriever:
            retrieved_results = retriever.retrieve(query, k=args.k)
            retrieved_docs = [doc['content'] for doc in retrieved_results]
            
            if args.verbose:
                print(f"Retrieved {len(retrieved_docs)} documents")
        
        # Generate answer
        if retrieved_docs:
            # RAG mode
            prompt = PromptManager.get_rag_prompt(args.dataset, query, retrieved_docs)
        else:
            # No-RAG mode
            prompt = PromptManager.get_no_rag_prompt(query)
        
        answer = llm.generate(prompt)
        
        if args.verbose:
            print(f"Answer: {answer}")
            print(f"Reference: {reference}")
        
        # Store results
        result = {
            'query_id': query_item['query_id'],
            'query': query,
            'predicted_answer': answer,
            'reference_answer': reference,
            'retrieved_docs': retrieved_docs
        }
        results.append(result)
        predictions.append(answer)
        references.append(reference)
    
    # Evaluation
    print("\n📊 Evaluating results...")
    evaluation_results = {}
    
    # NLP metrics
    nlp_evaluator = NLPMetrics(metrics=args.eval_metrics)
    nlp_scores = nlp_evaluator.evaluate(predictions, references)
    evaluation_results.update(nlp_scores)
    
    # LLM judge if requested
    if args.use_llm_judge:
        print("Using LLM as judge...")
        llm_judge = LLMJudge()
        judge_scores = llm_judge.evaluate(predictions, references)
        evaluation_results.update(judge_scores)
    
    # Print results
    print("\n" + "=" * 80)
    print("Evaluation Results:")
    print("=" * 80)
    for metric, score in evaluation_results.items():
        if metric != 'llm_judge_scores':  # Skip individual scores
            print(f"{metric:20s}: {score:.4f}")
    
    # Save results if requested
    if args.save:
        output_dir = os.path.join(
            args.output_dir, 
            args.dataset, 
            args.model,
            args.retrieval_method
        )
        os.makedirs(output_dir, exist_ok=True)
        
        # Save responses
        response_file = os.path.join(output_dir, f"responses_k{args.k}.json")
        with open(response_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save scores
        scores_file = os.path.join(output_dir, f"scores_k{args.k}.json")
        with open(scores_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2)
        
        print(f"\n💾 Results saved to {output_dir}")
    
    return evaluation_results


def main():
    """Main entry point"""
    parser = _build_parser()
    args = parser.parse_args()
    
    # Load config file if provided
    if args.config:
        try:
            config = ConfigManager.load_config(args.config)
            args = ConfigManager.merge_config_with_args(config, args, parser)
            print(f"📋 Loaded config from {args.config}")
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            return
    
    # Validate required arguments
    if not args.dataset:
        print("Error: --dataset is required (or specify in config file)")
        return
    if not args.model:
        print("Error: --model is required (or specify in config file)")
        return
    if args.retrieval_method == "hypercube" and not args.hypercube_file:
        print("Error: --hypercube_file is required when using hypercube retrieval method")
        return
    
    # Check environment variables
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    
    # Run pipeline
    try:
        run_pipeline(args)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()