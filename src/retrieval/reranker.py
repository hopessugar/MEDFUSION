"""Cross-encoder reranking module using NCBI MedCPT-Cross-Encoder for MedFusion-RAFT.

Reranks candidate retrieved passages by scoring query-passage pairs jointly,
capturing fine-grained cross-attention interactions.
"""

import logging
from typing import Any, List, Optional

from pydantic import BaseModel, Field

from src.models.schemas import RetrievedPassage

logger = logging.getLogger(__name__)


class RerankerConfig(BaseModel):
    """Configuration for cross-encoder reranking."""

    model_id: str = Field(
        default="ncbi/MedCPT-Cross-Encoder",
        description="HuggingFace model ID or local path for cross-encoder reranker",
    )
    top_k: int = Field(default=10, ge=1, description="Number of top reranked passages to return")
    batch_size: int = Field(default=16, ge=1, description="Batch size for pair scoring")
    max_length: int = Field(default=512, ge=64, description="Maximum combined sequence length")
    device: Optional[str] = Field(default=None, description="Torch device ('cuda', 'cpu', or None for auto)")


class CrossEncoderReranker:
    """Reranker utilizing MedCPT Cross-Encoder for deep query-passage relevance scoring."""

    def __init__(self, config: Optional[RerankerConfig] = None) -> None:
        """Initialize CrossEncoderReranker.

        Args:
            config: Reranker configuration parameters.
        """
        self.config = config or RerankerConfig()
        self._tokenizer: Optional[Any] = None
        self._model: Optional[Any] = None
        self._device: Optional[str] = None

    def load_model(self) -> None:
        """Load HuggingFace cross-encoder model and tokenizer."""
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as err:
            logger.error("torch and transformers are required for CrossEncoderReranker: %s", err)
            raise

        if self.config.device:
            self._device = self.config.device
        else:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading cross-encoder %s on device: %s", self.config.model_id, self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.config.model_id).to(self._device)
        self._model.eval()
        logger.info("Cross-encoder model loaded successfully")

    def rerank(self, query: str, passages: List[RetrievedPassage]) -> List[RetrievedPassage]:
        """Rerank a list of candidate passages against the query using cross-attention.

        Args:
            query: Medical query string.
            passages: Candidate retrieved passages.

        Returns:
            List of top_k passages reranked and updated with cross-encoder scores.
        """
        if not passages:
            return []

        if self._model is None or self._tokenizer is None:
            self.load_model()

        import torch

        pairs = [[query, p.text] for p in passages]
        all_scores: List[float] = []

        for i in range(0, len(pairs), self.config.batch_size):
            batch_pairs = pairs[i : i + self.config.batch_size]
            encoded = self._tokenizer(
                batch_pairs,
                truncation=True,
                padding=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                logits = self._model(**encoded).logits
                if logits.shape[-1] == 1:
                    batch_scores = logits.squeeze(-1).cpu().tolist()
                else:
                    # Multi-class / binary classification: take positive class probability
                    probs = torch.softmax(logits, dim=-1)
                    batch_scores = probs[:, 1].cpu().tolist()

            if isinstance(batch_scores, float):
                all_scores.append(batch_scores)
            else:
                all_scores.extend(batch_scores)

        # Pair up with original passages
        scored_pairs = list(zip(passages, all_scores))
        scored_pairs.sort(key=lambda item: item[1], reverse=True)

        k = min(self.config.top_k, len(scored_pairs))
        top_pairs = scored_pairs[:k]

        reranked_results: List[RetrievedPassage] = []
        for rank, (passage, score) in enumerate(top_pairs, start=1):
            method = (
                f"{passage.retrieval_method}_reranked"
                if not passage.retrieval_method.endswith("_reranked")
                else passage.retrieval_method
            )
            meta = passage.metadata.copy()
            meta["initial_retrieval_score"] = passage.score
            meta["initial_rank"] = passage.rank
            meta["reranker_score"] = float(score)

            reranked_results.append(
                RetrievedPassage(
                    passage_id=passage.passage_id,
                    doc_id=passage.doc_id,
                    text=passage.text,
                    metadata=meta,
                    section=passage.section,
                    score=float(score),
                    retrieval_method=method,
                    rank=rank,
                )
            )

        logger.debug("Reranked %d passages; returning top %d", len(passages), len(reranked_results))
        return reranked_results


class MockReranker:
    """Mock reranker for CPU unit tests and offline environments.

    Computes lexical term overlap scores as a fast deterministic stand-in.
    """

    def __init__(self, config: Optional[RerankerConfig] = None) -> None:
        """Initialize MockReranker.

        Args:
            config: RerankerConfig instance.
        """
        self.config = config or RerankerConfig()

    def load_model(self) -> None:
        """No-op for mock reranker."""
        logger.info("Initialized MockReranker")

    def rerank(self, query: str, passages: List[RetrievedPassage]) -> List[RetrievedPassage]:
        """Rerank passages using token overlap and prior score.

        Args:
            query: Query string.
            passages: Candidate passages.

        Returns:
            Reranked passages with updated scores and methods.
        """
        if not passages:
            return []

        q_terms = set(query.lower().split())

        def mock_score(p: RetrievedPassage) -> float:
            p_terms = set(p.text.lower().split())
            intersection = len(q_terms.intersection(p_terms))
            overlap = intersection / max(len(q_terms), 1)
            # Combine overlap with existing score to ensure differentiability
            return overlap + (p.score * 0.1)

        scored = [(p, mock_score(p)) for p in passages]
        scored.sort(key=lambda x: x[1], reverse=True)

        k = min(self.config.top_k, len(scored))
        results: List[RetrievedPassage] = []

        for rank, (passage, score) in enumerate(scored[:k], start=1):
            method = (
                f"{passage.retrieval_method}_reranked"
                if not passage.retrieval_method.endswith("_reranked")
                else passage.retrieval_method
            )
            meta = passage.metadata.copy()
            meta["mock_rerank_score"] = score

            results.append(
                RetrievedPassage(
                    passage_id=passage.passage_id,
                    doc_id=passage.doc_id,
                    text=passage.text,
                    metadata=meta,
                    section=passage.section,
                    score=float(score),
                    retrieval_method=method,
                    rank=rank,
                )
            )

        return results
