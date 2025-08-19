"""
Evaluation metrics for QA systems
"""

import os
import re
import string
from typing import List, Dict, Any
from abc import ABC, abstractmethod
import numpy as np
from collections import Counter


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
    """Traditional NLP metrics (BLEU, ROUGE, Exact Match, F1)"""
    
    def __init__(self, metrics: List[str] = None):
        """
        Initialize NLP metrics evaluator
        
        Args:
            metrics: List of metrics to compute. Default: all
        """
        self.metrics = metrics or ['exact_match', 'f1', 'bleu', 'rouge']
        
    def evaluate(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Compute NLP metrics
        
        Args:
            predictions: List of predicted answers
            references: List of reference answers
            
        Returns:
            Dictionary of metric scores
        """
        if len(predictions) != len(references):
            raise ValueError("Number of predictions and references must match")
        
        results = {}
        
        if 'exact_match' in self.metrics:
            results['exact_match'] = self._compute_exact_match(predictions, references)
        
        if 'f1' in self.metrics:
            results['f1'] = self._compute_f1(predictions, references)
        
        if 'bleu' in self.metrics:
            results['bleu'] = self._compute_bleu(predictions, references)
        
        if 'rouge' in self.metrics:
            rouge_scores = self._compute_rouge(predictions, references)
            results.update(rouge_scores)
        
        return results
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        return text.lower().strip().replace(".", "").replace(",", "").replace("?", "").replace("!", "")
    
    def _compute_exact_match(self, predictions: List[str], references: List[str]) -> float:
        """Compute exact match accuracy"""
        correct = 0
        for pred, ref in zip(predictions, references):
            if self._normalize_text(pred) == self._normalize_text(ref):
                correct += 1
        return correct / len(predictions) if predictions else 0
    
    def _compute_f1(self, predictions: List[str], references: List[str]) -> float:
        """Compute token-level F1 score"""
        f1_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = set(self._normalize_text(pred).split())
            ref_tokens = set(self._normalize_text(ref).split())
            
            if len(pred_tokens) == 0 or len(ref_tokens) == 0:
                f1_scores.append(0)
                continue
            
            common = pred_tokens & ref_tokens
            precision = len(common) / len(pred_tokens) if pred_tokens else 0
            recall = len(common) / len(ref_tokens) if ref_tokens else 0
            
            if precision + recall == 0:
                f1_scores.append(0)
            else:
                f1_scores.append(2 * precision * recall / (precision + recall))
        
        return np.mean(f1_scores) if f1_scores else 0
    
    def _compute_bleu(self, predictions: List[str], references: List[str]) -> float:
        """Compute BLEU score"""
        try:
            from nltk.translate.bleu_score import sentence_bleu
            
            bleu_scores = []
            for pred, ref in zip(predictions, references):
                pred_tokens = self._normalize_text(pred).split()
                ref_tokens = self._normalize_text(ref).split()
                
                if len(pred_tokens) == 0:
                    bleu_scores.append(0)
                else:
                    score = sentence_bleu([ref_tokens], pred_tokens, weights=(1, 0, 0, 0))
                    bleu_scores.append(score)
            
            return np.mean(bleu_scores) if bleu_scores else 0
            
        except ImportError:
            print("Warning: NLTK not available, skipping BLEU computation")
            return 0
    
    def _compute_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute ROUGE scores"""
        try:
            from rouge_score import rouge_scorer
            
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            
            rouge1_scores = []
            rouge2_scores = []
            rougel_scores = []
            
            for pred, ref in zip(predictions, references):
                scores = scorer.score(ref, pred)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                rougel_scores.append(scores['rougeL'].fmeasure)
            
            return {
                'rouge1': np.mean(rouge1_scores) if rouge1_scores else 0,
                'rouge2': np.mean(rouge2_scores) if rouge2_scores else 0,
                'rougeL': np.mean(rougel_scores) if rougel_scores else 0
            }
            
        except ImportError:
            print("Warning: rouge-score not available, skipping ROUGE computation")
            return {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}


class LLMJudge(Evaluator):
    """LLM-as-judge evaluation"""
    
    def __init__(self, judge_model: str = "gpt-4o", criteria: str = "accuracy"):
        """
        Initialize LLM judge
        
        Args:
            judge_model: Model to use as judge
            criteria: Evaluation criteria
        """
        self.judge_model = judge_model
        self.criteria = criteria
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
    
    def evaluate(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Use LLM to judge answer quality
        
        Args:
            predictions: List of predicted answers
            references: List of reference answers
            
        Returns:
            Dictionary with average score
        """
        scores = []
        
        for pred, ref in zip(predictions, references):
            score = self._judge_single(pred, ref)
            scores.append(score)
        
        return {
            'llm_judge_score': np.mean(scores) if scores else 0,
            'llm_judge_scores': scores  # Keep individual scores
        }
    
    def _judge_single(self, prediction: str, reference: str) -> float:
        """Judge a single prediction"""
        prompt = f"""
        You are evaluating the quality of an answer compared to a reference answer.
        
        Reference Answer: {reference}
        Predicted Answer: {prediction}
        
        Evaluation Criteria: {self.criteria}
        
        Rate the predicted answer on a scale from 0 to 1, where:
        - 0 means completely incorrect or irrelevant
        - 0.5 means partially correct
        - 1 means fully correct and complete
        
        Consider:
        1. Factual accuracy
        2. Completeness
        3. Relevance
        
        Output only a number between 0 and 1, nothing else.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=10
            )
            
            score_text = response.choices[0].message.content.strip()
            score = float(score_text)
            
            # Ensure score is in valid range
            return max(0, min(1, score))
            
        except Exception as e:
            print(f"Error in LLM judge: {e}")
            return 0.5  # Default to middle score on error