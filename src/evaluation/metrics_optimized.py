"""
Optimized evaluation metrics for QA systems
"""

import os
import re
import string
from typing import List, Dict, Any, Tuple, Optional
from abc import ABC, abstractmethod
import numpy as np
from collections import Counter
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor


class Evaluator(ABC):
    """Abstract base class for evaluation metrics"""
    
    @abstractmethod
    def evaluate(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Evaluate predictions against references
        
        Args:
            predictions: List of predicted answers
            references: List of reference answers
            
        Returns:
            Dictionary of metric scores
        """
        pass


class NLPMetrics(Evaluator):
    """Optimized NLP metrics (BLEU, ROUGE, Exact Match, F1)"""
    
    def __init__(self, metrics: List[str] = None):
        """
        Initialize NLP metrics evaluator
        
        Args:
            metrics: List of metrics to compute. Default: all
        """
        self.metrics = metrics or ['exact_match', 'f1', 'bleu', 'rouge', 'semantic']
        
        # Precompile regex patterns
        self._punct_pattern = re.compile(r'[.,;!?]+')
        self._space_pattern = re.compile(r'\s+')
        
        # Cache for normalized texts
        self._norm_cache = {}
        
        # Lazy load heavy dependencies
        self._nltk_loaded = False
        self._rouge_scorer = None
        self._semantic_model = None
        
    @lru_cache(maxsize=1024)
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison with caching"""
        # Remove punctuation and normalize spaces
        normalized = self._punct_pattern.sub(' ', text.lower())
        normalized = self._space_pattern.sub(' ', normalized).strip()
        return normalized
    
    def _tokenize(self, text: str) -> List[str]:
        """Fast tokenization"""
        return self._normalize_text(text).split()
    
    def evaluate(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Compute NLP metrics with parallel processing
        
        Args:
            predictions: List of predicted answers
            references: List of reference answers
            
        Returns:
            Dictionary of metric scores
        """
        if len(predictions) != len(references):
            raise ValueError("Number of predictions and references must match")
        
        results = {}
        
        # Precompute normalized texts for all metrics
        norm_preds = [self._normalize_text(p) for p in predictions]
        norm_refs = [self._normalize_text(r) for r in references]
        
        if 'exact_match' in self.metrics:
            results['exact_match'] = self._compute_exact_match_fast(norm_preds, norm_refs)
        
        if 'f1' in self.metrics:
            results['f1'] = self._compute_f1_fast(norm_preds, norm_refs)
        
        if 'bleu' in self.metrics:
            results['bleu'] = self._compute_bleu_fast(predictions, references)
        
        if 'rouge' in self.metrics:
            rouge_scores = self._compute_rouge_fast(predictions, references)
            results.update(rouge_scores)
        
        if 'semantic' in self.metrics:
            results['semantic'] = self._compute_semantic_fast(predictions, references)
        
        return results
    
    def _compute_exact_match_fast(self, norm_preds: List[str], norm_refs: List[str]) -> float:
        """Compute exact match accuracy using numpy"""
        matches = np.array([p == r for p, r in zip(norm_preds, norm_refs)])
        return np.mean(matches) if len(matches) > 0 else 0
    
    def _compute_f1_fast(self, norm_preds: List[str], norm_refs: List[str]) -> float:
        """Compute token-level F1 score with optimized set operations"""
        f1_scores = np.zeros(len(norm_preds))
        
        for i, (pred, ref) in enumerate(zip(norm_preds, norm_refs)):
            pred_tokens = set(pred.split())
            ref_tokens = set(ref.split())
            
            if not pred_tokens or not ref_tokens:
                continue
            
            common = pred_tokens & ref_tokens
            if not common:
                continue
            
            precision = len(common) / len(pred_tokens)
            recall = len(common) / len(ref_tokens)
            f1_scores[i] = 2 * precision * recall / (precision + recall)
        
        return np.mean(f1_scores)
    
    def _compute_bleu_fast(self, predictions: List[str], references: List[str]) -> float:
        """Compute BLEU score with batch processing"""
        if not self._nltk_loaded:
            try:
                import nltk
                # Download quietly if needed
                try:
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    nltk.download('punkt', quiet=True)
                self._nltk_loaded = True
            except ImportError:
                return 0
        
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        smoothing = SmoothingFunction().method1
        
        # Batch process with numpy
        bleu_scores = np.zeros(len(predictions))
        
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            pred_tokens = self._tokenize(pred)
            ref_tokens = self._tokenize(ref)
            
            if not pred_tokens:
                continue
            
            # Use smoothing to avoid zero scores
            bleu_scores[i] = sentence_bleu(
                [ref_tokens], pred_tokens, 
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smoothing
            )
        
        return np.mean(bleu_scores)
    
    def _compute_rouge_fast(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute ROUGE scores with cached scorer"""
        try:
            if self._rouge_scorer is None:
                from rouge_score import rouge_scorer
                self._rouge_scorer = rouge_scorer.RougeScorer(
                    ['rouge1', 'rouge2', 'rougeL'], 
                    use_stemmer=False  # Faster without stemming
                )
            
            # Use numpy arrays for efficiency
            rouge1 = np.zeros(len(predictions))
            rouge2 = np.zeros(len(predictions))
            rougel = np.zeros(len(predictions))
            
            for i, (pred, ref) in enumerate(zip(predictions, references)):
                scores = self._rouge_scorer.score(ref, pred)
                rouge1[i] = scores['rouge1'].fmeasure
                rouge2[i] = scores['rouge2'].fmeasure
                rougel[i] = scores['rougeL'].fmeasure
            
            return {
                'rouge1': np.mean(rouge1),
                'rouge2': np.mean(rouge2),
                'rougeL': np.mean(rougel)
            }
            
        except ImportError:
            return {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    
    def _compute_semantic_fast(self, predictions: List[str], references: List[str]) -> float:
        """Compute semantic similarity with batch encoding"""
        try:
            if self._semantic_model is None:
                from sentence_transformers import SentenceTransformer
                # Use smaller model for speed
                self._semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Batch encode all texts at once
            all_texts = predictions + references
            embeddings = self._semantic_model.encode(
                all_texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            pred_embs = embeddings[:len(predictions)]
            ref_embs = embeddings[len(predictions):]
            
            # Compute cosine similarities using numpy
            similarities = np.sum(pred_embs * ref_embs, axis=1) / (
                np.linalg.norm(pred_embs, axis=1) * np.linalg.norm(ref_embs, axis=1)
            )
            
            return np.mean(similarities)
            
        except ImportError:
            return 0


class AsyncLLMJudge(Evaluator):
    """Optimized asynchronous LLM-as-judge evaluation"""
    
    def __init__(self, judge_model: str = "gpt-4o-mini", criteria: str = "accuracy", 
                 batch_size: int = 5, max_retries: int = 3):
        """
        Initialize LLM judge with batching support
        
        Args:
            judge_model: Model to use as judge (use mini for speed)
            criteria: Evaluation criteria
            batch_size: Number of parallel requests
            max_retries: Maximum retries for failed requests
        """
        self.judge_model = judge_model
        self.criteria = criteria
        self.batch_size = batch_size
        self.max_retries = max_retries
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
    
    def evaluate(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Use LLM to judge answer quality with batch processing
        
        Args:
            predictions: List of predicted answers
            references: List of reference answers
            
        Returns:
            Dictionary with average score
        """
        # Process in batches for efficiency
        scores = []
        
        for i in range(0, len(predictions), self.batch_size):
            batch_preds = predictions[i:i+self.batch_size]
            batch_refs = references[i:i+self.batch_size]
            
            # Process batch in parallel
            with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
                batch_scores = list(executor.map(
                    self._judge_single_with_retry,
                    batch_preds,
                    batch_refs
                ))
            
            scores.extend(batch_scores)
        
        return {
            'llm_judge_score': np.mean(scores) if scores else 0,
            'llm_judge_scores': scores
        }
    
    def _judge_single_with_retry(self, prediction: str, reference: str) -> float:
        """Judge with retry logic"""
        for attempt in range(self.max_retries):
            try:
                return self._judge_single(prediction, reference)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    print(f"Failed to judge after {self.max_retries} attempts: {e}")
                    return 0.5  # Default neutral score
                continue
    
    def _judge_single(self, prediction: str, reference: str) -> float:
        """Judge a single prediction with optimized prompt"""
        # Shorter prompt for faster processing
        prompt = f"""Rate the predicted answer compared to reference (0-1 scale):
Reference: {reference[:500]}
Predicted: {prediction[:500]}
Score (0-1):"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0,
                n=1
            )
            
            # Extract score from response
            content = response.choices[0].message.content.strip()
            score = float(re.findall(r'[0-9.]+', content)[0])
            return min(max(score, 0), 1)  # Clamp to [0, 1]
            
        except Exception:
            return 0.5  # Default neutral score