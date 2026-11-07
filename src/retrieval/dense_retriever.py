"""Dense retrieval module using NCBI MedCPT encoders and FAISS for MedFusion-RAFT.

Provides MedCPTRetriever for production inference with NCBI MedCPT encoders,
and MockDenseRetriever for fast testing without GPU or network dependencies.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field

from src.models.schemas import Passage, RetrievedPassage

logger = logging.getLogger(__name__)

# Try importing FAISS with graceful numpy fallback
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    logger.info("faiss not installed; using built-in NumPy vector index fallback.")


class _NumpyVectorIndex:
    """NumPy-based Flat Inner Product index as fallback when FAISS is unavailable."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.vectors: Optional[np.ndarray] = None

    def add(self, vectors: np.ndarray) -> None:
        """Add vectors of shape (N, dim)."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])

    def search(self, query_vectors: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search query_vectors of shape (Q, dim). Returns (distances, indices)."""
        if self.vectors is None or len(self.vectors) == 0:
            return np.zeros((len(query_vectors), 0), dtype=np.float32), np.zeros((len(query_vectors), 0), dtype=np.int64)

        query_vectors = np.asarray(query_vectors, dtype=np.float32)
        # Compute dot products: (Q, dim) @ (dim, N) -> (Q, N)
        scores = np.dot(query_vectors, self.vectors.T)
        k = min(top_k, self.vectors.shape[0])
        # Get top-k indices per query
        indices = np.argsort(-scores, axis=1)[:, :k]
        distances = np.take_along_axis(scores, indices, axis=1)
        return distances.astype(np.float32), indices.astype(np.int64)

    @property
    def ntotal(self) -> int:
        return 0 if self.vectors is None else len(self.vectors)


class DenseRetrieverConfig(BaseModel):
    """Configuration for dense retrieval with MedCPT."""

    query_model_name_or_path: str = Field(
        default="ncbi/MedCPT-Query-Encoder",
        description="HuggingFace model ID or local path for query encoder",
    )
    article_model_name_or_path: str = Field(
        default="ncbi/MedCPT-Article-Encoder",
        description="HuggingFace model ID or local path for article/passage encoder",
    )
    embedding_dim: int = Field(default=768, description="Vector dimension of MedCPT representations")
    batch_size: int = Field(default=32, ge=1, description="Batch size for text encoding")
    top_k: int = Field(default=10, ge=1, description="Default number of passages to retrieve")
    index_type: str = Field(default="IndexFlatIP", description="FAISS index type (e.g. IndexFlatIP)")
    device: Optional[str] = Field(default=None, description="Torch device (cuda, cpu, or None for auto)")
    normalize_embeddings: bool = Field(default=True, description="Whether to L2-normalize embeddings")
    max_length: int = Field(default=512, ge=64, description="Maximum sequence token length")


class MedCPTRetriever:
    """Dense retriever using MedCPT query and article encoders with FAISS index."""

    def __init__(self, config: Optional[DenseRetrieverConfig] = None) -> None:
        """Initialize retriever with configuration."""
        self.config = config or DenseRetrieverConfig()
        self.passages: List[Passage] = []
        self.index: Optional[Any] = None
        self._query_tokenizer: Optional[Any] = None
        self._query_model: Optional[Any] = None
        self._article_tokenizer: Optional[Any] = None
        self._article_model: Optional[Any] = None
        self._device: Optional[str] = None

    def load_models(self) -> None:
        """Load HuggingFace tokenizers and models for query and article encoders."""
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as err:
            logger.error("torch or transformers is required for MedCPTRetriever: %s", err)
            raise

        if self.config.device:
            self._device = self.config.device
        else:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading MedCPT encoders on device: %s", self._device)
        self._query_tokenizer = AutoTokenizer.from_pretrained(self.config.query_model_name_or_path)
        self._query_model = AutoModel.from_pretrained(self.config.query_model_name_or_path).to(self._device)
        self._query_model.eval()

        self._article_tokenizer = AutoTokenizer.from_pretrained(self.config.article_model_name_or_path)
        self._article_model = AutoModel.from_pretrained(self.config.article_model_name_or_path).to(self._device)
        self._article_model.eval()
        logger.info("MedCPT encoders loaded successfully")

    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """Encode query strings using MedCPT Query Encoder.

        Args:
            queries: List of query texts.

        Returns:
            Normalized float32 numpy array of shape (len(queries), embedding_dim).
        """
        if self._query_model is None or self._query_tokenizer is None:
            self.load_models()

        import torch

        all_embeds: List[np.ndarray] = []
        for i in range(0, len(queries), self.config.batch_size):
            batch_texts = queries[i : i + self.config.batch_size]
            encoded = self._query_tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                outputs = self._query_model(**encoded)
                # MedCPT uses the [CLS] representation (token 0)
                embeds = outputs.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)
                all_embeds.append(embeds)

        embeddings = np.vstack(all_embeds) if all_embeds else np.zeros((0, self.config.embedding_dim), dtype=np.float32)
        if self.config.normalize_embeddings and len(embeddings) > 0:
            if HAS_FAISS:
                faiss.normalize_L2(embeddings)
            else:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / np.maximum(norms, 1e-12)
        return embeddings

    def encode_passages(self, passages: List[Passage]) -> np.ndarray:
        """Encode passages using MedCPT Article Encoder.

        Args:
            passages: List of Passage objects.

        Returns:
            Normalized float32 numpy array of shape (len(passages), embedding_dim).
        """
        if self._article_model is None or self._article_tokenizer is None:
            self.load_models()

        import torch

        texts = [p.text for p in passages]
        all_embeds: List[np.ndarray] = []

        for i in range(0, len(texts), self.config.batch_size):
            batch_texts = texts[i : i + self.config.batch_size]
            encoded = self._article_tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                outputs = self._article_model(**encoded)
                embeds = outputs.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)
                all_embeds.append(embeds)

        embeddings = np.vstack(all_embeds) if all_embeds else np.zeros((0, self.config.embedding_dim), dtype=np.float32)
        if self.config.normalize_embeddings and len(embeddings) > 0:
            if HAS_FAISS:
                faiss.normalize_L2(embeddings)
            else:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / np.maximum(norms, 1e-12)
        return embeddings

    def build_index(self, passages: List[Passage]) -> None:
        """Encode passages and build a FAISS inner-product index.

        Args:
            passages: List of passages to index.
        """
        if not passages:
            raise ValueError("Passage list cannot be empty when building index")

        self.passages = list(passages)
        logger.info("Encoding %d passages with MedCPT Article Encoder", len(self.passages))
        vectors = self.encode_passages(self.passages)

        dim = vectors.shape[1] if len(vectors) > 0 else self.config.embedding_dim
        if HAS_FAISS:
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(vectors)
        else:
            self.index = _NumpyVectorIndex(dim)
            self.index.add(vectors)

        logger.info("Dense index built successfully with %d entries", len(self.passages))

    def search(self, query: str, top_k: Optional[int] = None) -> List[RetrievedPassage]:
        """Retrieve most similar passages for a query using cosine similarity.

        Args:
            query: Query text.
            top_k: Number of passages to return. Defaults to config.top_k.

        Returns:
            List of RetrievedPassage objects ranked by cosine score.
        """
        if self.index is None or not self.passages:
            raise RuntimeError("Dense index has not been built. Call build_index first.")

        k = top_k if top_k is not None else self.config.top_k
        k = min(k, len(self.passages))

        q_emb = self.encode_queries([query])
        distances, indices = self.index.search(q_emb, k)

        results: List[RetrievedPassage] = []
        for rank, (score, idx) in enumerate(zip(distances[0], indices[0]), start=1):
            if idx < 0 or idx >= len(self.passages):
                continue
            p = self.passages[idx]
            results.append(
                RetrievedPassage(
                    passage_id=p.passage_id,
                    doc_id=p.doc_id,
                    text=p.text,
                    metadata=p.metadata.copy(),
                    section=p.section,
                    score=float(score),
                    retrieval_method="dense_medcpt",
                    rank=rank,
                )
            )

        return results

    def save_index(self, path: Union[str, Path]) -> None:
        """Save FAISS index and passage metadata to disk."""
        if self.index is None:
            raise RuntimeError("Index is uninitialized; cannot save.")

        target_dir = Path(path).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)

        idx_file = target_dir / "dense.index"
        meta_file = target_dir / "dense_passages.pkl"

        if HAS_FAISS and isinstance(self.index, faiss.Index):
            faiss.write_index(self.index, str(idx_file))
        else:
            with open(idx_file, "wb") as f:
                pickle.dump(self.index, f)

        with open(meta_file, "wb") as f:
            pickle.dump({
                "passages": [p.model_dump() for p in self.passages],
                "config": self.config.model_dump(),
            }, f)
        logger.info("Dense retriever saved to %s", target_dir)

    def load_index(self, path: Union[str, Path]) -> None:
        """Load FAISS index and passage metadata from disk.

        Security Note:
            Metadata and fallback indices are deserialized via `pickle`. Only load
            files from trusted and authenticated filesystem locations.
        """
        target_dir = Path(path).resolve()
        idx_file = target_dir / "dense.index"
        meta_file = target_dir / "dense_passages.pkl"

        if not idx_file.is_file() or not meta_file.is_file():
            raise FileNotFoundError(f"Missing dense index files in {target_dir}")

        if HAS_FAISS:
            try:
                self.index = faiss.read_index(str(idx_file))
            except Exception:
                with open(idx_file, "rb") as f:
                    self.index = pickle.load(f)
        else:
            with open(idx_file, "rb") as f:
                self.index = pickle.load(f)

        with open(meta_file, "rb") as f:
            data = pickle.load(f)

        self.passages = [Passage(**p) for p in data["passages"]]
        self.config = DenseRetrieverConfig(**data["config"])
        logger.info("Dense retriever loaded from %s with %d passages", target_dir, len(self.passages))


class MockDenseRetriever(MedCPTRetriever):
    """Deterministic mock dense retriever for CPU testing without model downloads."""

    def load_models(self) -> None:
        """Mock model loading - does not download or initialize PyTorch models."""
        logger.info("Initialized MockDenseRetriever (no external weights needed)")

    def _hash_to_vector(self, text: str, dim: int) -> np.ndarray:
        """Deterministic pseudo-embedding from text hash."""
        rng = np.random.RandomState(abs(hash(text)) % (2**31))
        vec = rng.randn(dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / max(norm, 1e-12)

    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """Encode query strings deterministically without models."""
        dim = self.config.embedding_dim
        vectors = [self._hash_to_vector(q, dim) for q in queries]
        return np.vstack(vectors) if vectors else np.zeros((0, dim), dtype=np.float32)

    def encode_passages(self, passages: List[Passage]) -> np.ndarray:
        """Encode passages deterministically without models."""
        dim = self.config.embedding_dim
        vectors = [self._hash_to_vector(p.text, dim) for p in passages]
        return np.vstack(vectors) if vectors else np.zeros((0, dim), dtype=np.float32)
