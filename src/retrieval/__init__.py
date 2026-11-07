"""Retrieval package for MedFusion-RAFT.

Exports sparse retrieval, dense retrieval, hybrid fusion, cross-encoder reranking,
end-to-end retrieval pipelines, and retrieval evaluation metrics.
"""

from src.retrieval.bm25_retriever import BM25Config, BM25Retriever
from src.retrieval.dense_retriever import (
    DenseRetrieverConfig,
    MedCPTRetriever,
    MockDenseRetriever,
)
from src.retrieval.evaluation import (
    EvaluationSummary,
    evaluate_retrieval,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src.retrieval.fusion import FusionConfig, HybridFusion
from src.retrieval.pipeline import RetrievalPipeline, RetrievalPipelineConfig
from src.retrieval.reranker import (
    CrossEncoderReranker,
    MockReranker,
    RerankerConfig,
)

__all__ = [
    # Sparse BM25
    "BM25Config",
    "BM25Retriever",
    # Dense MedCPT
    "DenseRetrieverConfig",
    "MedCPTRetriever",
    "MockDenseRetriever",
    # Fusion
    "FusionConfig",
    "HybridFusion",
    # Reranking
    "RerankerConfig",
    "CrossEncoderReranker",
    "MockReranker",
    # Pipeline
    "RetrievalPipelineConfig",
    "RetrievalPipeline",
    # Evaluation
    "recall_at_k",
    "precision_at_k",
    "mrr",
    "ndcg_at_k",
    "evaluate_retrieval",
    "EvaluationSummary",
]
