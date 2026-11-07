"""BM25 sparse retriever for MedFusion-RAFT.

Implements Okapi BM25 retrieval for medical text passages with medical-aware tokenization,
index serialization, and graceful fallback when rank_bm25 is not installed.
"""

import logging
import math
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from pydantic import BaseModel, Field

from src.models.schemas import Passage, RetrievedPassage

logger = logging.getLogger(__name__)

try:
    from rank_bm25 import BM25Okapi
    HAS_RANK_BM25 = True
except ImportError:
    HAS_RANK_BM25 = False
    logger.info("rank_bm25 not installed; using built-in BM25Okapi fallback implementation.")


class _PurePythonBM25:
    """Built-in pure-Python Okapi BM25 implementation as fallback."""

    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lens = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_lens) / self.corpus_size if self.corpus_size > 0 else 0.0

        # Term frequencies and document frequencies
        self.doc_freqs: Dict[str, int] = Counter()
        self.doc_tfs: List[Dict[str, int]] = []

        for doc in corpus:
            tf = Counter(doc)
            self.doc_tfs.append(tf)
            for token in tf.keys():
                self.doc_freqs[token] += 1

        # Precompute IDF with Robertson-Spärck Jones formula
        self.idf: Dict[str, float] = {}
        for term, freq in self.doc_freqs.items():
            self.idf[term] = math.log(((self.corpus_size - freq + 0.5) / (freq + 0.5)) + 1.0)

    def get_scores(self, query: Sequence[str]) -> List[float]:
        """Compute BM25 scores for all documents in the corpus for the given query."""
        scores = [0.0] * self.corpus_size
        for term in query:
            if term not in self.idf:
                continue
            term_idf = self.idf[term]
            for doc_idx, tf in enumerate(self.doc_tfs):
                cnt = tf.get(term, 0)
                if cnt > 0:
                    numerator = cnt * (self.k1 + 1.0)
                    denominator = cnt + self.k1 * (1.0 - self.b + self.b * (self.doc_lens[doc_idx] / self.avgdl))
                    scores[doc_idx] += term_idf * (numerator / denominator)
        return scores


class BM25Config(BaseModel):
    """Configuration for BM25 retriever."""

    k1: float = Field(default=1.5, description="Term frequency saturation parameter")
    b: float = Field(default=0.75, description="Document length normalization parameter")
    top_k: int = Field(default=10, ge=1, description="Default number of passages to retrieve")
    lowercase: bool = Field(default=True, description="Whether to lowercase tokens")
    strip_accents: bool = Field(default=True, description="Whether to normalize accents")


class BM25Retriever:
    """Sparse retrieval engine based on the Okapi BM25 algorithm."""

    # Matches alphanumeric tokens, preserving hyphens inside medical terms (e.g. covid-19)
    TOKEN_PATTERN = re.compile(r"\b[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*\b")

    def __init__(self, config: Optional[BM25Config] = None) -> None:
        """Initialize the BM25 retriever.

        Args:
            config: Optional BM25Config parameters.
        """
        self.config = config or BM25Config()
        self.passages: List[Passage] = []
        self._bm25: Optional[Any] = None

    def tokenize(self, text: str) -> List[str]:
        """Preprocess and tokenize medical text into normalized tokens.

        Args:
            text: Query or document string.

        Returns:
            List of normalized string tokens.
        """
        if self.config.lowercase:
            text = text.lower()
        tokens = self.TOKEN_PATTERN.findall(text)
        return tokens

    def build_index(self, passages: List[Passage]) -> None:
        """Build the BM25 index from a collection of Passage objects.

        Args:
            passages: List of Passage objects to index.

        Raises:
            ValueError: If passages list is empty.
        """
        if not passages:
            raise ValueError("Cannot build BM25 index from an empty passage list")

        self.passages = list(passages)
        tokenized_corpus = [self.tokenize(p.text) for p in self.passages]

        logger.info("Building BM25 index over %d passages", len(self.passages))
        if HAS_RANK_BM25:
            self._bm25 = BM25Okapi(tokenized_corpus, k1=self.config.k1, b=self.config.b)
        else:
            self._bm25 = _PurePythonBM25(tokenized_corpus, k1=self.config.k1, b=self.config.b)

        logger.info("BM25 index successfully built")

    def search(self, query: str, top_k: Optional[int] = None) -> List[RetrievedPassage]:
        """Search the indexed passages for the query using BM25 scoring.

        Args:
            query: Query string.
            top_k: Number of results to return. If None, uses config.top_k.

        Returns:
            List of RetrievedPassage objects sorted by descending BM25 score.

        Raises:
            RuntimeError: If search is invoked before build_index.
        """
        if self._bm25 is None or not self.passages:
            raise RuntimeError("BM25 index has not been built. Call build_index first.")

        k = top_k if top_k is not None else self.config.top_k
        k = min(k, len(self.passages))

        query_tokens = self.tokenize(query)
        if not query_tokens:
            logger.warning("Empty tokenized query for: %s", query)
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Get indices of top-k scores in descending order
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        results: List[RetrievedPassage] = []
        for rank, idx in enumerate(top_indices, start=1):
            passage = self.passages[idx]
            retrieved = RetrievedPassage(
                passage_id=passage.passage_id,
                doc_id=passage.doc_id,
                text=passage.text,
                metadata=passage.metadata.copy(),
                section=passage.section,
                score=float(scores[idx]),
                retrieval_method="bm25",
                rank=rank,
            )
            results.append(retrieved)

        return results

    def save_index(self, path: Union[str, Path]) -> None:
        """Serialize the BM25 model and passages to disk.

        Args:
            path: Destination file path.
        """
        if self._bm25 is None:
            raise RuntimeError("Cannot save an uninitialized BM25 index")

        file_path = Path(path).resolve()
        file_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "config": self.config.model_dump(),
            "passages": [p.model_dump() for p in self.passages],
            "bm25": self._bm25,
        }
        with open(file_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("BM25 index saved to %s", file_path)

    def load_index(self, path: Union[str, Path]) -> None:
        """Load a serialized BM25 index from disk.

        Args:
            path: Source file path.
        """
        file_path = Path(path).resolve()
        if not file_path.is_file():
            raise FileNotFoundError(f"Index file not found: {file_path}")

        with open(file_path, "rb") as f:
            data = pickle.load(f)

        self.config = BM25Config(**data["config"])
        self.passages = [Passage(**p) for p in data["passages"]]
        self._bm25 = data["bm25"]
        logger.info("BM25 index loaded from %s with %d passages", file_path, len(self.passages))
