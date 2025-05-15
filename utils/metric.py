import nltk
from nltk.tokenize import word_tokenize

# Download NLTK data for tokenization
nltk.download('punkt')


def exact_match(pred, gold):
    return int(pred.strip().lower() == gold.strip().lower())

def f1_score(pred, gold):
    pred_tokens = word_tokenize(pred.lower())
    gold_tokens = word_tokenize(gold.lower())
    common = set(pred_tokens) & set(gold_tokens)
    if len(common) == 0:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

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