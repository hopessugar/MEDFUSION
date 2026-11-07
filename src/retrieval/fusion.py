"""Hybrid fusion module implementing Reciprocal Rank Fusion (RRF) for MedFusion-RAFT.

Combines ranked result lists from diverse retrieval modalities (e.g., BM25 and MedCPT dense)
into a unified, deduplicated ranking.
"""

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.models.schemas import RetrievedPassage

logger = logging.getLogger(__name__)


class FusionConfig(BaseModel):
    """Configuration for result fusion."""

    method: str = Field(default="rrf", description="Fusion method to use ('rrf')")
    rrf_k: int = Field(
        default=60,
        ge=1,
        description="Reciprocal Rank Fusion smoothing parameter k (typically 60)",
    )
    top_k: int = Field(
        default=10,
        ge=1,
        description="Maximum number of merged passages to return",
    )


class HybridFusion:
    """Combines multiple candidate lists into a single ranked list using Reciprocal Rank Fusion."""

    def __init__(self, config: Optional[FusionConfig] = None) -> None:
        """Initialize HybridFusion with parameters.

        Args:
            config: FusionConfig instance.
        """
        self.config = config or FusionConfig()

    def fuse(
        self,
        results_lists: List[List[RetrievedPassage]],
        weights: Optional[List[float]] = None,
    ) -> List[RetrievedPassage]:
        """Fuse multiple ranked passage lists into a single deduplicated ranking.

        Score formula:
            score(d) = sum(w_i * (1.0 / (k + rank_i(d))))

        Args:
            results_lists: List of ranked candidate lists (e.g. [bm25_results, dense_results]).
            weights: Optional weighting factor per candidate list. If None, equal weights (1.0) are used.

        Returns:
            Deduplicated list of RetrievedPassage objects sorted by fused score.
        """
        if not results_lists:
            return []

        # Filter out empty candidate lists while retaining corresponding weights
        active_lists: List[List[RetrievedPassage]] = []
        active_weights: List[float] = []

        provided_weights = weights if weights is not None else [1.0] * len(results_lists)
        if len(provided_weights) != len(results_lists):
            raise ValueError(
                f"Weights length ({len(provided_weights)}) must match results_lists length ({len(results_lists)})"
            )

        for res_list, w in zip(results_lists, provided_weights):
            if res_list:
                active_lists.append(res_list)
                active_weights.append(float(w))

        if not active_lists:
            return []

        rrf_scores: Dict[str, float] = {}
        passage_store: Dict[str, RetrievedPassage] = {}
        provenance: Dict[str, Dict[str, Any]] = {}

        for list_idx, (r_list, weight) in enumerate(zip(active_lists, active_weights)):
            for rank_0, passage in enumerate(r_list):
                rank = rank_0 + 1  # 1-based rank
                pid = passage.passage_id
                rrf_term = weight * (1.0 / (self.config.rrf_k + rank))

                if pid not in rrf_scores:
                    rrf_scores[pid] = 0.0
                    passage_store[pid] = passage
                    provenance[pid] = {
                        "component_ranks": {},
                        "component_scores": {},
                    }

                rrf_scores[pid] += rrf_term
                method_key = passage.retrieval_method or f"list_{list_idx}"
                provenance[pid]["component_ranks"][method_key] = rank
                provenance[pid]["component_scores"][method_key] = passage.score

        # Sort passage IDs by descending fused RRF score
        sorted_pids = sorted(rrf_scores.keys(), key=lambda pid: rrf_scores[pid], reverse=True)
        top_pids = sorted_pids[: self.config.top_k]

        fused_results: List[RetrievedPassage] = []
        for rank, pid in enumerate(top_pids, start=1):
            base_passage = passage_store[pid]
            merged_metadata = base_passage.metadata.copy()
            merged_metadata["fusion_details"] = {
                "fused_score": rrf_scores[pid],
                "component_ranks": provenance[pid]["component_ranks"],
                "component_scores": provenance[pid]["component_scores"],
            }

            fused_results.append(
                RetrievedPassage(
                    passage_id=base_passage.passage_id,
                    doc_id=base_passage.doc_id,
                    text=base_passage.text,
                    metadata=merged_metadata,
                    section=base_passage.section,
                    score=float(rrf_scores[pid]),
                    retrieval_method="hybrid_rrf",
                    rank=rank,
                )
            )

        logger.debug(
            "Fused %d candidate lists into %d unique passages (top %d returned)",
            len(active_lists),
            len(rrf_scores),
            len(fused_results),
        )
        return fused_results
