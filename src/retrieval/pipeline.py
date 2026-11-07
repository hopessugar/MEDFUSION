"""Integrated retrieval pipeline for MedFusion-RAFT.

Orchestrates sparse BM25 retrieval, dense MedCPT vector retrieval, reciprocal rank
fusion, and cross-encoder reranking into an end-to-end medical retrieval pipeline.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field

from src.models.schemas import Passage, RetrievedPassage
from src.retrieval.bm25_retriever import BM25Config, BM25Retriever
from src.retrieval.dense_retriever import (
    DenseRetrieverConfig,
    MedCPTRetriever,
    MockDenseRetriever,
)
from src.retrieval.fusion import FusionConfig, HybridFusion
from src.retrieval.reranker import (
    CrossEncoderReranker,
    MockReranker,
    RerankerConfig,
)

logger = logging.getLogger(__name__)


class RetrievalPipelineConfig(BaseModel):
    """Configuration for end-to-end RetrievalPipeline."""

    enable_bm25: bool = Field(default=True, description="Enable BM25 sparse search")
    enable_dense: bool = Field(default=True, description="Enable MedCPT dense search")
    enable_fusion: bool = Field(default=True, description="Enable RRF fusion between sparse and dense")
    enable_reranker: bool = Field(default=True, description="Enable cross-encoder reranker")
    bm25_top_k: int = Field(default=20, ge=1, description="Number of candidates from BM25")
    dense_top_k: int = Field(default=20, ge=1, description="Number of candidates from dense search")
    final_top_k: int = Field(default=5, ge=1, description="Final number of retrieved passages")
    bm25_weight: float = Field(default=1.0, description="BM25 weight in fusion")
    dense_weight: float = Field(default=1.0, description="Dense search weight in fusion")
    use_mock_components: bool = Field(
        default=False,
        description="Whether to use MockDenseRetriever and MockReranker (for CPU testing)",
    )


class RetrievalPipeline:
    """Full-featured medical retrieval engine orchestrating ingestion, retrieval, and reranking."""

    def __init__(
        self,
        bm25_retriever: Optional[BM25Retriever] = None,
        dense_retriever: Optional[Union[MedCPTRetriever, MockDenseRetriever]] = None,
        fusion: Optional[HybridFusion] = None,
        reranker: Optional[Union[CrossEncoderReranker, MockReranker]] = None,
        config: Optional[RetrievalPipelineConfig] = None,
    ) -> None:
        """Initialize pipeline with injected components or defaults.

        Args:
            bm25_retriever: BM25 retriever instance.
            dense_retriever: Dense retriever instance.
            fusion: Hybrid fusion instance.
            reranker: Reranker instance.
            config: Pipeline configuration.
        """
        self.config = config or RetrievalPipelineConfig()

        # Injected or default BM25 retriever
        self.bm25_retriever = bm25_retriever or BM25Retriever(
            config=BM25Config(top_k=self.config.bm25_top_k)
        )

        # Injected or default Dense retriever
        if dense_retriever is not None:
            self.dense_retriever = dense_retriever
        else:
            dense_cfg = DenseRetrieverConfig(top_k=self.config.dense_top_k)
            if self.config.use_mock_components:
                self.dense_retriever = MockDenseRetriever(config=dense_cfg)
            else:
                self.dense_retriever = MedCPTRetriever(config=dense_cfg)

        # Injected or default Fusion
        self.fusion = fusion or HybridFusion(
            config=FusionConfig(top_k=max(self.config.bm25_top_k, self.config.dense_top_k))
        )

        # Injected or default Reranker
        if reranker is not None:
            self.reranker = reranker
        else:
            reranker_cfg = RerankerConfig(top_k=self.config.final_top_k)
            if self.config.use_mock_components:
                self.reranker = MockReranker(config=reranker_cfg)
            else:
                self.reranker = CrossEncoderReranker(config=reranker_cfg)

        logger.info(
            "Initialized RetrievalPipeline (bm25=%s, dense=%s, fusion=%s, reranker=%s, mock=%s)",
            self.config.enable_bm25,
            self.config.enable_dense,
            self.config.enable_fusion,
            self.config.enable_reranker,
            self.config.use_mock_components,
        )

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "RetrievalPipeline":
        """Construct RetrievalPipeline from a YAML configuration file.

        Args:
            yaml_path: Path to yaml config.

        Returns:
            Configured RetrievalPipeline instance.
        """
        path = Path(yaml_path).resolve()
        with open(path, "r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f) or {}

        pipeline_cfg = RetrievalPipelineConfig(**raw_cfg.get("pipeline", {}))
        bm25_cfg = BM25Config(**raw_cfg.get("bm25", {}))
        dense_cfg = DenseRetrieverConfig(**raw_cfg.get("dense", {}))
        fusion_cfg = FusionConfig(**raw_cfg.get("fusion", {}))
        reranker_cfg = RerankerConfig(**raw_cfg.get("reranker", {}))

        bm25 = BM25Retriever(config=bm25_cfg)
        dense = (
            MockDenseRetriever(config=dense_cfg)
            if pipeline_cfg.use_mock_components
            else MedCPTRetriever(config=dense_cfg)
        )
        fusion = HybridFusion(config=fusion_cfg)
        reranker = (
            MockReranker(config=reranker_cfg)
            if pipeline_cfg.use_mock_components
            else CrossEncoderReranker(config=reranker_cfg)
        )

        return cls(
            bm25_retriever=bm25,
            dense_retriever=dense,
            fusion=fusion,
            reranker=reranker,
            config=pipeline_cfg,
        )

    def build_index(self, passages: List[Passage]) -> None:
        """Build indices for both sparse and dense retrieval.

        Args:
            passages: List of Passage objects to index.
        """
        if not passages:
            raise ValueError("Passages list is empty; cannot build retrieval indices")

        logger.info("Building indices over %d passages", len(passages))
        t0 = time.perf_counter()

        if self.config.enable_bm25:
            bm25_t0 = time.perf_counter()
            self.bm25_retriever.build_index(passages)
            logger.info("BM25 index built in %.3fs", time.perf_counter() - bm25_t0)

        if self.config.enable_dense:
            dense_t0 = time.perf_counter()
            self.dense_retriever.build_index(passages)
            logger.info("Dense index built in %.3fs", time.perf_counter() - dense_t0)

        logger.info("Total index build time: %.3fs", time.perf_counter() - t0)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedPassage]:
        """Execute end-to-end retrieval and reranking for a query.

        Steps:
            1. BM25 search (if enabled)
            2. Dense search (if enabled)
            3. RRF fusion (if multiple modalities enabled)
            4. Cross-encoder reranking (if enabled)
            5. Return top_k candidates

        Args:
            query: Question or query string.
            top_k: Number of final passages to return. Defaults to config.final_top_k.

        Returns:
            List of final RetrievedPassage objects.
        """
        final_k = top_k if top_k is not None else self.config.final_top_k
        query = query.strip()
        if not query:
            logger.warning("Empty query received in retrieve()")
            return []

        results_lists: List[List[RetrievedPassage]] = []
        weights: List[float] = []
        timings: Dict[str, float] = {}

        # 1. BM25 Search
        if self.config.enable_bm25:
            t0 = time.perf_counter()
            bm25_results = self.bm25_retriever.search(query, top_k=self.config.bm25_top_k)
            timings["bm25_search_ms"] = (time.perf_counter() - t0) * 1000.0
            results_lists.append(bm25_results)
            weights.append(self.config.bm25_weight)

        # 2. Dense Search
        if self.config.enable_dense:
            t0 = time.perf_counter()
            dense_results = self.dense_retriever.search(query, top_k=self.config.dense_top_k)
            timings["dense_search_ms"] = (time.perf_counter() - t0) * 1000.0
            results_lists.append(dense_results)
            weights.append(self.config.dense_weight)

        # 3. Fusion
        if self.config.enable_fusion and len(results_lists) > 1:
            t0 = time.perf_counter()
            fused = self.fusion.fuse(results_lists, weights=weights)
            timings["fusion_ms"] = (time.perf_counter() - t0) * 1000.0
            candidates = fused
        elif results_lists:
            candidates = results_lists[0]
        else:
            logger.warning("No retrieval modalities enabled in pipeline")
            return []

        # 4. Cross-Encoder Reranking
        if self.config.enable_reranker and self.reranker is not None and candidates:
            t0 = time.perf_counter()
            reranked = self.reranker.rerank(query, candidates)
            timings["reranking_ms"] = (time.perf_counter() - t0) * 1000.0
            candidates = reranked

        # Final truncation to desired top_k
        final_results = candidates[:final_k]

        # Attach pipeline stage timings to top passages' metadata for observability
        for p in final_results:
            p.metadata["pipeline_timings_ms"] = timings

        logger.info(
            "Query '%s' returned %d passages. Timings: %s",
            query[:40],
            len(final_results),
            timings,
        )
        return final_results

    def save_indices(self, dir_path: Union[str, Path]) -> None:
        """Save sparse and dense indices to a directory.

        Args:
            dir_path: Target directory.
        """
        target = Path(dir_path).resolve()
        target.mkdir(parents=True, exist_ok=True)

        if self.config.enable_bm25:
            self.bm25_retriever.save_index(target / "bm25.pkl")

        if self.config.enable_dense:
            self.dense_retriever.save_index(target / "dense")

        logger.info("All pipeline indices saved into %s", target)

    def load_indices(self, dir_path: Union[str, Path]) -> None:
        """Load sparse and dense indices from a directory.

        Args:
            dir_path: Source directory.
        """
        source = Path(dir_path).resolve()

        if self.config.enable_bm25 and (source / "bm25.pkl").is_file():
            self.bm25_retriever.load_index(source / "bm25.pkl")

        if self.config.enable_dense and (source / "dense").is_dir():
            self.dense_retriever.load_index(source / "dense")

        logger.info("All pipeline indices loaded from %s", source)
