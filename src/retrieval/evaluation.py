"""Retrieval evaluation metrics module for MedFusion-RAFT.

Computes standard information retrieval metrics: Recall@K, Precision@K, MRR,
and NDCG@K for assessing retriever effectiveness.
"""

import logging
import math
from typing import Any, Dict, List, Sequence, Set, Union

from pydantic import BaseModel, Field

from src.models.schemas import RetrievedPassage

logger = logging.getLogger(__name__)


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Union[Set[str], Sequence[str]],
    k: int,
) -> float:
    """Calculate Recall@K.

    Recall@K = (Number of relevant passages in top K) / (Total relevant passages)

    Args:
        retrieved_ids: Ordered sequence of retrieved passage IDs.
        relevant_ids: Set or list of gold relevant passage IDs.
        k: Cutoff rank.

    Returns:
        Recall value in [0.0, 1.0].
    """
    rel_set = set(relevant_ids)
    if not rel_set or k <= 0:
        return 0.0

    top_k = retrieved_ids[:k]
    hits = sum(1 for pid in top_k if pid in rel_set)
    return hits / float(len(rel_set))


def precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Union[Set[str], Sequence[str]],
    k: int,
) -> float:
    """Calculate Precision@K.

    Precision@K = (Number of relevant passages in top K) / K

    Args:
        retrieved_ids: Ordered sequence of retrieved passage IDs.
        relevant_ids: Set or list of gold relevant passage IDs.
        k: Cutoff rank.

    Returns:
        Precision value in [0.0, 1.0].
    """
    if k <= 0:
        return 0.0

    rel_set = set(relevant_ids)
    top_k = retrieved_ids[:k]
    hits = sum(1 for pid in top_k if pid in rel_set)
    return hits / float(k)


def mrr(
    retrieved_ids: Sequence[str],
    relevant_ids: Union[Set[str], Sequence[str]],
) -> float:
    """Calculate Mean Reciprocal Rank (MRR) for a single query.

    MRR = 1 / rank of first relevant item (1-based), or 0.0 if not found.

    Args:
        retrieved_ids: Ordered sequence of retrieved passage IDs.
        relevant_ids: Set or list of gold relevant passage IDs.

    Returns:
        Reciprocal rank score in [0.0, 1.0].
    """
    rel_set = set(relevant_ids)
    if not rel_set:
        return 0.0

    for rank, pid in enumerate(retrieved_ids, start=1):
        if pid in rel_set:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Union[Set[str], Sequence[str]],
    k: int,
) -> float:
    """Calculate Normalized Discounted Cumulative Gain at K (NDCG@K) with binary relevance.

    DCG@K = sum_{i=1}^K (rel_i / log2(i + 1))
    IDCG@K = sum_{i=1}^{min(K, |rel|)} (1 / log2(i + 1))
    NDCG@K = DCG@K / IDCG@K

    Args:
        retrieved_ids: Ordered sequence of retrieved passage IDs.
        relevant_ids: Set or list of gold relevant passage IDs.
        k: Cutoff rank.

    Returns:
        NDCG score in [0.0, 1.0].
    """
    rel_set = set(relevant_ids)
    if not rel_set or k <= 0:
        return 0.0

    # Compute DCG@k
    dcg = 0.0
    for i, pid in enumerate(retrieved_ids[:k], start=1):
        if pid in rel_set:
            dcg += 1.0 / math.log2(i + 1)

    # Compute Ideal DCG@k
    idcg = 0.0
    num_ideal = min(k, len(rel_set))
    for i in range(1, num_ideal + 1):
        idcg += 1.0 / math.log2(i + 1)

    return (dcg / idcg) if idcg > 0.0 else 0.0


class EvaluationSummary(BaseModel):
    """Pydantic model containing aggregate retrieval evaluation metrics."""

    num_queries: int = Field(..., description="Number of evaluated queries")
    metrics: Dict[str, float] = Field(..., description="Averaged metric scores")


def evaluate_retrieval(
    queries: Sequence[str],
    retrieved_results: Sequence[Sequence[RetrievedPassage]],
    ground_truth: Sequence[Union[Set[str], Sequence[str]]],
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> Dict[str, float]:
    """Evaluate retrieval results across multiple queries and compute aggregated metrics.

    Args:
        queries: List of query strings.
        retrieved_results: List of retrieved passage lists for each query.
        ground_truth: List of relevant passage IDs for each query.
        k_values: Cutoff ranks to evaluate (e.g. [1, 3, 5, 10]).

    Returns:
        Dictionary mapping metric names (e.g. 'Recall@5', 'MRR') to averaged float scores.

    Raises:
        ValueError: If length of retrieved_results or ground_truth doesn't match queries.
    """
    n = len(queries)
    if len(retrieved_results) != n or len(ground_truth) != n:
        raise ValueError(
            f"Input lengths do not match: queries={n}, retrieved={len(retrieved_results)}, truth={len(ground_truth)}"
        )

    if n == 0:
        logger.warning("Empty query evaluation set passed to evaluate_retrieval")
        return {"num_queries": 0.0}

    # Metric accumulators
    mrr_total = 0.0
    recall_totals: Dict[int, float] = {k: 0.0 for k in k_values}
    precision_totals: Dict[int, float] = {k: 0.0 for k in k_values}
    ndcg_totals: Dict[int, float] = {k: 0.0 for k in k_values}

    for res, truth in zip(retrieved_results, ground_truth):
        retrieved_ids = [p.passage_id for p in res]
        truth_set = set(truth)

        mrr_total += mrr(retrieved_ids, truth_set)

        for k in k_values:
            recall_totals[k] += recall_at_k(retrieved_ids, truth_set, k)
            precision_totals[k] += precision_at_k(retrieved_ids, truth_set, k)
            ndcg_totals[k] += ndcg_at_k(retrieved_ids, truth_set, k)

    summary: Dict[str, float] = {
        "num_queries": float(n),
        "MRR": round(mrr_total / n, 4),
    }

    for k in sorted(k_values):
        summary[f"Recall@{k}"] = round(recall_totals[k] / n, 4)
        summary[f"Precision@{k}"] = round(precision_totals[k] / n, 4)
        summary[f"NDCG@{k}"] = round(ndcg_totals[k] / n, 4)

    logger.info("Evaluation completed over %d queries: %s", n, summary)
    return summary
