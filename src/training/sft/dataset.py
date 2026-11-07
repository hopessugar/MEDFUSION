"""SFT dataset preparation and formatting for MedFusion-RAFT."""

from __future__ import annotations

import hashlib
import random
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.models.schemas import AnswerabilityLabel, TrainingSFTExample
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Optional HF datasets import handled gracefully
try:
    from datasets import Dataset, DatasetDict, load_dataset

    DATASETS_AVAILABLE = True
except ImportError:
    Dataset = None  # type: ignore[assignment]
    DatasetDict = None  # type: ignore[assignment]
    load_dataset = None  # type: ignore[assignment]
    DATASETS_AVAILABLE = False

DEFAULT_SYSTEM_PROMPT = (
    "You are a medical question answering assistant. Ground your answers in the provided evidence. "
    "Cite passage IDs. If evidence is insufficient, say so."
)


class SFTDatasetConfig(BaseModel):
    """Configuration for SFT dataset construction and formatting.

    Attributes:
        max_seq_length: Maximum sequence length in tokens.
        template_style: Prompt template style ('chat', 'gemma', 'raw').
        include_system_prompt: Whether to prepend the medical system prompt.
        system_prompt: System prompt instructing the model on clinical QA and grounding.
        seed: Random seed for deterministic shuffling and splitting.
    """

    max_seq_length: int = Field(default=2048, ge=64, description="Maximum token sequence length")
    template_style: str = Field(default="chat", description="Chat template style")
    include_system_prompt: bool = Field(
        default=True, description="Whether to include system prompt"
    )
    system_prompt: str = Field(
        default=DEFAULT_SYSTEM_PROMPT, description="Medical assistant system prompt"
    )
    seed: int = Field(default=42, description="Random seed for data splitting")


class SFTDatasetBuilder:
    """Builder for constructing, formatting, and splitting SFT datasets."""

    def __init__(
        self,
        config: Optional[SFTDatasetConfig] = None,
        tokenizer: Optional[Any] = None,
    ) -> None:
        """Initialize SFTDatasetBuilder.

        Args:
            config: Dataset preparation configuration.
            tokenizer: Optional Hugging Face tokenizer for length calculation.
        """
        self.config = config or SFTDatasetConfig()
        self.tokenizer = tokenizer

    def format_for_training(self, example: TrainingSFTExample) -> str:
        """Format an SFT example into a chat-aligned prompt-completion text.

        The format follows MedGemma's standard conversational turn structure:
        System: medical assistant guidance
        User: clinical question and evidence context
        Assistant: structured response with citations and answerability

        Args:
            example: TrainingSFTExample instance.

        Returns:
            Single string representing formatted training sample.
        """
        citations_str = ", ".join(example.citation_ids) if example.citation_ids else "None"
        answerability_str = (
            example.answerability.value
            if isinstance(example.answerability, AnswerabilityLabel)
            else str(example.answerability)
        )

        user_content = (
            f"Question: {example.question.strip()}\n\n"
            f"Evidence:\n{example.context.strip()}\n\n"
            "Instructions:\n"
            "1. Answer based ONLY on the provided evidence.\n"
            "2. Cite specific passage IDs for each claim.\n"
            "3. If evidence is insufficient, state INSUFFICIENT_EVIDENCE."
        )

        assistant_content = (
            f"ANSWER: {example.answer.strip()}\n"
            f"CITATIONS: {citations_str}\n"
            f"ANSWERABILITY: {answerability_str}\n"
            "CONFIDENCE: 0.95"
        )

        if self.config.template_style == "gemma":
            parts = []
            if self.config.include_system_prompt:
                parts.append(
                    f"<start_of_turn>user\n{self.config.system_prompt}\n\n{user_content}<end_of_turn>"
                )
            else:
                parts.append(f"<start_of_turn>user\n{user_content}<end_of_turn>")
            parts.append(f"<start_of_turn>model\n{assistant_content}<end_of_turn>")
            return "\n".join(parts)

        # Default conversational format
        formatted = ""
        if self.config.include_system_prompt:
            formatted += f"System: {self.config.system_prompt}\n\n"
        formatted += f"User: {user_content}\n\n"
        formatted += f"Assistant: {assistant_content}"
        return formatted

    def to_chat_messages(self, example: TrainingSFTExample) -> list[dict[str, str]]:
        """Convert an SFT example to a list of message dicts for apply_chat_template.

        Args:
            example: TrainingSFTExample instance.

        Returns:
            List of message dictionaries with 'role' and 'content'.
        """
        citations_str = ", ".join(example.citation_ids) if example.citation_ids else "None"
        answerability_str = (
            example.answerability.value
            if isinstance(example.answerability, AnswerabilityLabel)
            else str(example.answerability)
        )

        user_content = (
            f"Question: {example.question.strip()}\n\n"
            f"Evidence:\n{example.context.strip()}"
        )
        assistant_content = (
            f"ANSWER: {example.answer.strip()}\n"
            f"CITATIONS: {citations_str}\n"
            f"ANSWERABILITY: {answerability_str}\n"
            "CONFIDENCE: 0.95"
        )

        messages = []
        if self.config.include_system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": assistant_content})
        return messages

    def from_examples(
        self,
        examples: list[TrainingSFTExample],
        doc_ids: Optional[list[str]] = None,
    ) -> Any:
        """Create a HuggingFace Dataset from a list of TrainingSFTExamples.

        Args:
            examples: List of TrainingSFTExample instances.
            doc_ids: Optional list of document IDs parallel to examples for document splitting.

        Returns:
            Hugging Face Dataset instance (or dict if datasets is unavailable).
        """
        records: list[dict[str, Any]] = []
        for i, ex in enumerate(examples):
            text = self.format_for_training(ex)
            doc_id = doc_ids[i] if doc_ids and i < len(doc_ids) else f"doc_{i}"
            records.append(
                {
                    "text": text,
                    "question": ex.question,
                    "context": ex.context,
                    "answer": ex.answer,
                    "citation_ids": ex.citation_ids,
                    "answerability": (
                        ex.answerability.value
                        if isinstance(ex.answerability, AnswerabilityLabel)
                        else str(ex.answerability)
                    ),
                    "doc_id": doc_id,
                }
            )

        if not DATASETS_AVAILABLE or Dataset is None:
            logger.warning("HF datasets not installed; returning dictionary representation.")
            return {"data": records}

        return Dataset.from_list(records)

    def from_pubmedqa(self, split: str = "train", limit: Optional[int] = None) -> Any:
        """Load and transform the PubMedQA dataset into SFT format.

        Args:
            split: Dataset split ('train', 'validation', 'test').
            limit: Optional cap on the number of records to load.

        Returns:
            Formatted Hugging Face Dataset.

        Raises:
            RuntimeError: If datasets package is not installed.
        """
        if not DATASETS_AVAILABLE or load_dataset is None:
            raise RuntimeError("The 'datasets' library is required to load PubMedQA.")

        logger.info("Loading PubMedQA dataset", split=split, limit=limit)
        hf_dataset = load_dataset("pubmed_qa", "pqa_labeled", split=split)
        if limit is not None:
            hf_dataset = hf_dataset.select(range(min(limit, len(hf_dataset))))

        sft_examples: list[TrainingSFTExample] = []
        doc_ids: list[str] = []

        for row in hf_dataset:
            question = row.get("question", "")
            context_data = row.get("context", {})
            contexts = context_data.get("contexts", [])
            labels = context_data.get("labels", [])

            context_parts = []
            citation_ids = []
            for idx, text in enumerate(contexts):
                p_id = f"P{idx + 1}"
                label = labels[idx] if idx < len(labels) else f"Section {idx + 1}"
                context_parts.append(f"[{p_id}] ({label}): {text}")
                citation_ids.append(p_id)

            full_context = "\n".join(context_parts)
            long_answer = row.get("long_answer", "")
            final_decision = row.get("final_decision", "yes")

            answer_text = f"{long_answer.strip()} (Conclusion: {final_decision.upper()})"
            pubid = str(row.get("pubid", f"pubmed_{len(doc_ids)}"))

            example = TrainingSFTExample(
                question=question,
                context=full_context,
                answer=answer_text,
                citation_ids=citation_ids,
                answerability=AnswerabilityLabel.ANSWERABLE,
            )
            sft_examples.append(example)
            doc_ids.append(pubid)

        logger.info("Processed PubMedQA examples", count=len(sft_examples))
        return self.from_examples(sft_examples, doc_ids=doc_ids)

    def from_medqa(self, split: str = "train", limit: Optional[int] = None) -> Any:
        """Load and transform MedQA USMLE dataset into SFT format.

        Args:
            split: Dataset split ('train', 'validation', 'test').
            limit: Optional cap on number of records.

        Returns:
            Formatted Hugging Face Dataset.

        Raises:
            RuntimeError: If datasets package is not installed.
        """
        if not DATASETS_AVAILABLE or load_dataset is None:
            raise RuntimeError("The 'datasets' library is required to load MedQA.")

        logger.info("Loading MedQA dataset", split=split, limit=limit)
        try:
            hf_dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split=split)
        except Exception:
            # Fallback to local or alternative path
            logger.warning("Could not load GBaker/MedQA-USMLE-4-options directly; trying med_qa")
            hf_dataset = load_dataset("med_qa", split=split)

        if limit is not None:
            hf_dataset = hf_dataset.select(range(min(limit, len(hf_dataset))))

        sft_examples: list[TrainingSFTExample] = []
        doc_ids: list[str] = []

        for idx, row in enumerate(hf_dataset):
            question = row.get("question", "")
            options = row.get("options", {})
            answer_idx = row.get("answer_idx", "")
            answer_val = row.get("answer", "")

            # Format options into evidence context
            opt_lines = []
            for opt_key, opt_text in (options.items() if isinstance(options, dict) else []):
                opt_lines.append(f"[{opt_key}] {opt_text}")
            evidence_context = (
                "\n".join(opt_lines) if opt_lines else "Clinical diagnostic context provided above."
            )

            answer_text = (
                f"Option ({answer_idx}) is correct: {answer_val}"
                if answer_idx
                else str(answer_val)
            )

            example = TrainingSFTExample(
                question=question,
                context=evidence_context,
                answer=answer_text,
                citation_ids=[answer_idx] if answer_idx else ["P1"],
                answerability=AnswerabilityLabel.ANSWERABLE,
            )
            sft_examples.append(example)
            doc_ids.append(f"medqa_{idx}")

        logger.info("Processed MedQA examples", count=len(sft_examples))
        return self.from_examples(sft_examples, doc_ids=doc_ids)

    def validate_dataset(self, dataset: Any) -> dict[str, Any]:
        """Perform data quality and schema validation on an SFT dataset.

        Args:
            dataset: HuggingFace Dataset or list-of-dicts representation.

        Returns:
            Validation report dictionary with metrics and potential quality issues.
        """
        items = dataset if isinstance(dataset, list) else list(dataset)
        total_count = len(items)
        empty_texts = 0
        missing_citations = 0
        answerability_dist: dict[str, int] = {}
        total_chars = 0
        exceeds_max_len = 0

        for row in items:
            text = row.get("text", "")
            if not text or not str(text).strip():
                empty_texts += 1

            chars = len(str(text))
            total_chars += chars
            # Rough token estimate (chars / 4)
            approx_tokens = chars // 4
            if approx_tokens > self.config.max_seq_length:
                exceeds_max_len += 1

            ans_status = str(row.get("answerability", "unknown"))
            answerability_dist[ans_status] = answerability_dist.get(ans_status, 0) + 1

            cites = row.get("citation_ids", [])
            if not cites:
                missing_citations += 1

        avg_chars = (total_chars / total_count) if total_count > 0 else 0.0

        report = {
            "total_samples": total_count,
            "empty_samples": empty_texts,
            "missing_citations_count": missing_citations,
            "answerability_distribution": answerability_dist,
            "avg_char_length": round(avg_chars, 2),
            "estimated_exceeding_max_len": exceeds_max_len,
            "is_valid": (empty_texts == 0 and total_count > 0),
        }
        logger.info("SFT Dataset validation completed", **report)
        return report

    def split_dataset(
        self,
        dataset: Any,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        doc_id_key: str = "doc_id",
    ) -> dict[str, Any]:
        """Split a dataset into train/validation/test sets at the document level.

        Grouping by document identifier guarantees that passages or QA pairs from the same
        source document never appear across split boundaries, preventing data leakage.

        Args:
            dataset: Hugging Face Dataset or list of records.
            train_ratio: Proportion of documents allocated to training.
            val_ratio: Proportion of documents allocated to validation.
            test_ratio: Proportion of documents allocated to test.
            doc_id_key: Key in records representing document ID.

        Returns:
            Dictionary with keys 'train', 'validation', 'test' containing split Datasets.

        Raises:
            ValueError: If split ratios do not sum approximately to 1.0.
        """
        total_ratio = train_ratio + val_ratio + test_ratio
        if not (0.99 <= total_ratio <= 1.01):
            raise ValueError(f"Split ratios must sum to 1.0, got: {total_ratio}")

        items = dataset if isinstance(dataset, list) else [row for row in dataset]

        # Group sample indices by document ID
        doc_to_indices: dict[str, list[int]] = {}
        for idx, row in enumerate(items):
            doc_id = str(row.get(doc_id_key) or "")
            if not doc_id:
                # Deterministic hash fallback on question content
                q_text = str(row.get("question", f"sample_{idx}"))
                doc_id = hashlib.md5(q_text.encode("utf-8")).hexdigest()[:12]
            doc_to_indices.setdefault(doc_id, []).append(idx)

        unique_docs = sorted(list(doc_to_indices.keys()))
        rng = random.Random(self.config.seed)
        rng.shuffle(unique_docs)

        n_docs = len(unique_docs)
        n_train = int(round(n_docs * train_ratio))
        n_val = int(round(n_docs * val_ratio))

        train_docs = set(unique_docs[:n_train])
        val_docs = set(unique_docs[n_train : n_train + n_val])
        test_docs = set(unique_docs[n_train + n_val :])

        train_indices: list[int] = []
        val_indices: list[int] = []
        test_indices: list[int] = []

        for doc_id, indices in doc_to_indices.items():
            if doc_id in train_docs:
                train_indices.extend(indices)
            elif doc_id in val_docs:
                val_indices.extend(indices)
            else:
                test_indices.extend(indices)

        logger.info(
            "Document-level split complete",
            total_docs=n_docs,
            train_docs=len(train_docs),
            val_docs=len(val_docs),
            test_docs=len(test_docs),
            train_samples=len(train_indices),
            val_samples=len(val_indices),
            test_samples=len(test_indices),
        )

        if DATASETS_AVAILABLE and Dataset is not None and isinstance(dataset, Dataset):
            return {
                "train": dataset.select(train_indices),
                "validation": dataset.select(val_indices),
                "test": dataset.select(test_indices),
            }

        return {
            "train": [items[i] for i in train_indices],
            "validation": [items[i] for i in val_indices],
            "test": [items[i] for i in test_indices],
        }
