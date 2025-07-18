import nltk
from nltk.tokenize import word_tokenize
from collections import Counter

import re
import string


# Download NLTK data for tokenization
nltk.download('punkt')


def exact_match(pred, gold):
    return int(pred.strip().lower() == gold.strip().lower())

# def f1_score2(pred, gold):
#     pred_tokens = word_tokenize(pred.lower())
#     gold_tokens = word_tokenize(gold.lower())
#     common = set(pred_tokens) & set(gold_tokens)
#     if len(common) == 0:
#         return 0.0
#     precision = len(common) / len(pred_tokens)
#     recall = len(common) / len(gold_tokens)
#     return 2 * precision * recall / (precision + recall)



def normalize_answer(answer: str) -> str:
    """
    Normalize a given string by applying the following transformations:
    1. Convert the string to lowercase.
    2. Remove punctuation characters.
    3. Remove the articles "a", "an", and "the".
    4. Normalize whitespace by collapsing multiple spaces into one.

    Args:
        answer (str): The input string to be normalized.

    Returns:
        str: The normalized string.
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(answer))))


def f1_score(gold: str, predicted: str) -> float:
    gold_tokens = normalize_answer(gold).split()
    predicted_tokens = normalize_answer(predicted).split()
    common = Counter(predicted_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(predicted_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    return 2 * (precision * recall) / (precision + recall)


def bleu_score(pred, gold):
    """
    It shows word overlap between new answer and ground-truth. It penalizes rephrasing, even if the meaning is correct.
    """
    from nltk.translate.bleu_score import sentence_bleu

    score = sentence_bleu([pred.split()], gold.split())
    return score

def rouge_score(pred, gold):
    """
    ROUGE-L focuses on longest common subsequences, good for long sentences, as it captures structural similarity. But Less sensitive to semantic meaning.
    """
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(pred, gold)
    return scores['rougeL'].fmeasure

def semantic_score(pred, gold):
    """
    Meaning similarity using contextual embeddings (e.g., from BERT or SentenceTransformers).
    Higher scores mean the new answer is semantically closer to the ground-truth, even if worded differently. 
    Captures paraphrasing and meaning better than BLEU/ROUGE.
    """
    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode([pred, gold])
    similarity = util.cos_sim(embeddings[0], embeddings[1])

    return similarity.item()


def precision_at_k(retrieved_docs, relevant_docs):
    relevant_retrieved = [doc for doc in retrieved_docs if doc in relevant_docs]
    return len(relevant_retrieved) / len(retrieved_docs)

def recall_at_k(retrieved_docs, relevant_docs):
    relevant_retrieved = [doc for doc in retrieved_docs if doc in relevant_docs]
    return len(relevant_retrieved) / len(relevant_docs) if relevant_docs else 0.0
