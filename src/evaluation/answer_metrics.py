"""Answer quality evaluation metrics for MedFusion-RAFT.

Computes text-level precision, recall, F1, Exact Match, ROUGE-L, and BLEU
between generated medical responses and gold-standard clinical references.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Sequence

from src.models.schemas import DatasetExample, PipelineResult
from src.utils.logging import get_logger

logger = get_logger(__name__)


def normalize_text(text: str) -> str:
    """Normalize text by lowercasing, removing punctuation, articles, and extra whitespace.

    Args:
        text: Input raw string.

    Returns:
        Normalized string representation.
    """
    text = text.lower()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = "".join(ch for ch in text if ch not in string.punctuation)
    # Normalize whitespace
    return " ".join(text.split())


def _get_tokens(text: str) -> list[str]:
    """Tokenize normalized text into word tokens."""
    return normalize_text(text).split()


def compute_exact_match(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Compute Exact Match (EM) score over prediction and reference pairs.

    Args:
        predictions: List of candidate answer strings.
        references: List of gold standard target strings.

    Returns:
        Fraction of exact normalized matches in [0.0, 1.0].
    """
    if not predictions or not references:
        return 0.0

    n = min(len(predictions), len(references))
    if n == 0:
        return 0.0

    matches = sum(
        1.0 for p, r in zip(predictions[:n], references[:n]) if normalize_text(p) == normalize_text(r)
    )
    return matches / n


def _token_f1_single(prediction: str, reference: str) -> float:
    """Compute token-level F1 score for a single prediction and reference pair."""
    pred_tokens = _get_tokens(prediction)
    ref_tokens = _get_tokens(reference)

    if not pred_tokens or not ref_tokens:
        return 1.0 if pred_tokens == ref_tokens else 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_f1(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Compute macro-averaged token-level F1 score across all pairs.

    Args:
        predictions: List of candidate answers.
        references: List of ground-truth target answers.

    Returns:
        Mean token F1 score in [0.0, 1.0].
    """
    if not predictions or not references:
        return 0.0

    n = min(len(predictions), len(references))
    if n == 0:
        return 0.0

    scores = [_token_f1_single(p, r) for p, r in zip(predictions[:n], references[:n])]
    return sum(scores) / n


def compute_accuracy(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Compute clinical QA accuracy based on containment, option match, or exact match.

    Args:
        predictions: List of candidate predictions.
        references: List of reference answers.

    Returns:
        Clinical accuracy score in [0.0, 1.0].
    """
    if not predictions or not references:
        return 0.0

    n = min(len(predictions), len(references))
    if n == 0:
        return 0.0

    correct = 0
    for pred, ref in zip(predictions[:n], references[:n]):
        norm_pred = normalize_text(pred)
        norm_ref = normalize_text(ref)

        if not norm_ref:
            continue

        # Check exact equality or containment
        if norm_pred == norm_ref or norm_ref in norm_pred:
            correct += 1
        elif _token_f1_single(pred, ref) >= 0.70:
            correct += 1

    return correct / n


def _longest_common_subsequence(seq1: list[str], seq2: list[str]) -> int:
    """Calculate the length of the Longest Common Subsequence between two token sequences."""
    m, n = len(seq1), len(seq2)
    if m == 0 or n == 0:
        return 0

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if seq1[i] == seq2[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    return dp[m][n]


def compute_rouge_l(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Compute ROUGE-L F1 score based on longest common subsequence.

    Args:
        predictions: List of candidate answers.
        references: List of gold answers.

    Returns:
        Mean ROUGE-L score in [0.0, 1.0].
    """
    if not predictions or not references:
        return 0.0

    n = min(len(predictions), len(references))
    if n == 0:
        return 0.0

    scores: list[float] = []
    for pred, ref in zip(predictions[:n], references[:n]):
        p_tokens = _get_tokens(pred)
        r_tokens = _get_tokens(ref)

        if not p_tokens or not r_tokens:
            scores.append(1.0 if p_tokens == r_tokens else 0.0)
            continue

        lcs = _longest_common_subsequence(p_tokens, r_tokens)
        if lcs == 0:
            scores.append(0.0)
            continue

        prec = lcs / len(p_tokens)
        rec = lcs / len(r_tokens)
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        scores.append(f1)

    return sum(scores) / n


def compute_bleu(
    predictions: Sequence[str], references: Sequence[str], max_n: int = 4
) -> float:
    """Compute BLEU score with modified n-gram precision and brevity penalty.

    Args:
        predictions: List of candidate answers.
        references: List of reference answers.
        max_n: Maximum n-gram order (defaults to 4).

    Returns:
        Mean BLEU score in [0.0, 1.0].
    """
    if not predictions or not references:
        return 0.0

    n = min(len(predictions), len(references))
    if n == 0:
        return 0.0

    sentence_bleus: list[float] = []
    for pred, ref in zip(predictions[:n], references[:n]):
        p_tokens = _get_tokens(pred)
        r_tokens = _get_tokens(ref)

        if not p_tokens or not r_tokens:
            sentence_bleus.append(1.0 if p_tokens == r_tokens else 0.0)
            continue

        # Brevity penalty
        bp = 1.0
        if len(p_tokens) < len(r_tokens):
            import math

            bp = math.exp(1 - (len(r_tokens) / len(p_tokens))) if len(p_tokens) > 0 else 0.0

        precisions: list[float] = []
        for i in range(1, max_n + 1):
            if len(p_tokens) < i or len(r_tokens) < i:
                precisions.append(0.0)
                continue

            p_ngrams = Counter(tuple(p_tokens[j : j + i]) for j in range(len(p_tokens) - i + 1))
            r_ngrams = Counter(tuple(r_tokens[j : j + i]) for j in range(len(r_tokens) - i + 1))

            overlap = sum((p_ngrams & r_ngrams).values())
            total = sum(p_ngrams.values())
            precisions.append((overlap + 0.1) / (total + 0.1) if total > 0 else 0.0)

        import math

        log_sum = sum(math.log(p) for p in precisions if p > 0)
        score = bp * math.exp(log_sum / max_n) if precisions else 0.0
        sentence_bleus.append(score)

    return sum(sentence_bleus) / n


def evaluate_answers(
    results: list[PipelineResult], ground_truth: list[DatasetExample]
) -> dict[str, float]:
    """Evaluate end-to-end question answering performance across multiple metrics.

    Args:
        results: List of execution results from MedFusionOrchestrator.
        ground_truth: Matching list of benchmark ground-truth examples.

    Returns:
        Dictionary of calculated answer quality metrics.
    """
    gt_map = {ex.question_id: ex for ex in ground_truth}

    preds: list[str] = []
    refs: list[str] = []

    for r in results:
        qid = r.question.question_id
        ex = gt_map.get(qid)
        if ex is None:
            continue

        # Determine prediction text
        pred_text = r.final_answer or (
            r.generated_answer.answer_text if r.generated_answer else ""
        )
        preds.append(pred_text)
        refs.append(ex.reference_answer)

    if not preds:
        logger.warning("No matching question IDs found between results and ground truth")
        return {
            "accuracy": 0.0,
            "f1": 0.0,
            "exact_match": 0.0,
            "rouge_l": 0.0,
            "bleu": 0.0,
        }

    acc = compute_accuracy(preds, refs)
    f1 = compute_f1(preds, refs)
    em = compute_exact_match(preds, refs)
    rouge = compute_rouge_l(preds, refs)
    bleu = compute_bleu(preds, refs)

    metrics = {
        "accuracy": round(acc, 4),
        "f1": round(f1, 4),
        "exact_match": round(em, 4),
        "rouge_l": round(rouge, 4),
        "bleu": round(bleu, 4),
    }

    logger.info("Answer evaluation completed over %d examples: %s", len(preds), metrics)
    return metrics
