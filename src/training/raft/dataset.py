"""RAFT dataset preparation and formatting for MedFusion-RAFT.

Implements the Retrieval-Augmented Fine-Tuning (RAFT) data construction strategy
where each training example is composed of a question, a mix of gold and distractor
passages, and a target answer that cites only the correct passage IDs.

Reference: Zhang et al., "RAFT: Adapting Language Model to Domain Specific RAG" (2024).
"""

from __future__ import annotations

import random
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.models.schemas import (
    AnswerabilityLabel,
    Passage,
    TrainingRAFTExample,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Optional HF datasets import
try:
    from datasets import Dataset

    DATASETS_AVAILABLE = True
except ImportError:
    Dataset = None  # type: ignore[assignment]
    DATASETS_AVAILABLE = False

RAFT_SYSTEM_PROMPT = (
    "You are a medical question answering assistant. Answer questions based on "
    "the provided evidence. Cite passage IDs for each claim. If evidence is "
    "insufficient, state INSUFFICIENT_EVIDENCE."
)


class RAFTDatasetConfig(BaseModel):
    """Configuration for RAFT dataset construction.

    Attributes:
        distractor_ratio: Fraction of training examples that are distractor-only
            (gold passage removed). Must be in [0.0, 1.0].
        num_distractor_passages: Number of distractor (irrelevant) passages to
            include per training example alongside the gold passage.
        include_unanswerable: Whether to generate unanswerable examples where
            no gold passage is present.
        max_seq_length: Maximum token length for formatted training text.
        template_style: Chat template style ('gemma', 'chat', 'raw').
        system_prompt: System-level prompt prepended to each example.
        seed: Random seed for reproducibility.
        shuffle_passages: Whether to shuffle passage order in each example.
    """

    distractor_ratio: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Fraction of examples that are distractor-only",
    )
    num_distractor_passages: int = Field(
        default=3, ge=0,
        description="Number of distractor passages per example",
    )
    include_unanswerable: bool = Field(
        default=True,
        description="Generate unanswerable training examples",
    )
    max_seq_length: int = Field(
        default=2048, ge=64,
        description="Maximum token sequence length",
    )
    template_style: str = Field(
        default="gemma",
        description="Chat template style",
    )
    system_prompt: str = Field(
        default=RAFT_SYSTEM_PROMPT,
        description="Medical assistant system prompt",
    )
    seed: int = Field(default=42, description="Random seed for reproducibility")
    shuffle_passages: bool = Field(
        default=True,
        description="Whether to shuffle passage order in each example",
    )


class RAFTDatasetBuilder:
    """Builder for constructing RAFT training datasets.

    Given a set of medical QA examples and a passage pool, constructs training
    instances following the RAFT methodology:
    - Answerable examples: gold passage + distractor passages
    - Unanswerable examples: distractor passages only (gold removed)
    - Mixed examples with controlled distractor ratios
    """

    def __init__(
        self,
        config: Optional[RAFTDatasetConfig] = None,
        passage_pool: Optional[list[Passage]] = None,
    ) -> None:
        """Initialize RAFTDatasetBuilder.

        Args:
            config: RAFT dataset configuration. Uses defaults if None.
            passage_pool: Pool of passages available for distractor sampling.
                If None, distractors are randomly selected from other examples.
        """
        self.config = config or RAFTDatasetConfig()
        self.passage_pool = passage_pool or []
        self._rng = random.Random(self.config.seed)
        self._category_counts: dict[str, int] = {
            "answerable": 0,
            "unanswerable": 0,
            "distractor_heavy": 0,
        }

    @property
    def category_distribution(self) -> dict[str, int]:
        """Return the current distribution of generated example categories."""
        return dict(self._category_counts)

    def _select_distractors(
        self,
        gold_passage_ids: set[str],
        num_distractors: int,
        exclude_doc_ids: Optional[set[str]] = None,
    ) -> list[Passage]:
        """Select distractor passages from the passage pool.

        Args:
            gold_passage_ids: IDs of gold passages to exclude.
            num_distractors: Number of distractors to select.
            exclude_doc_ids: Optional doc_ids to exclude (to prevent
                near-duplicate distractors from the same document).

        Returns:
            List of distractor Passage objects.
        """
        exclude_doc_ids = exclude_doc_ids or set()
        candidates = [
            p for p in self.passage_pool
            if p.passage_id not in gold_passage_ids
            and p.doc_id not in exclude_doc_ids
        ]

        if len(candidates) < num_distractors:
            # If pool is too small, allow doc_id overlap but still exclude gold
            candidates = [
                p for p in self.passage_pool
                if p.passage_id not in gold_passage_ids
            ]

        if len(candidates) <= num_distractors:
            return list(candidates)

        return self._rng.sample(candidates, num_distractors)

    def _format_passages(
        self,
        passages: list[Passage],
        shuffle: bool = True,
    ) -> tuple[str, dict[str, str]]:
        """Format passages with sequential IDs (P1, P2, ...).

        Args:
            passages: List of passages to format.
            shuffle: Whether to shuffle passage order.

        Returns:
            Tuple of (formatted passage text, mapping from passage_id to P-label).
        """
        indexed_passages = list(enumerate(passages))
        if shuffle:
            self._rng.shuffle(indexed_passages)

        formatted_parts: list[str] = []
        id_mapping: dict[str, str] = {}

        for new_idx, (_, passage) in enumerate(indexed_passages):
            p_label = f"P{new_idx + 1}"
            id_mapping[passage.passage_id] = p_label
            section_info = f" ({passage.section})" if passage.section else ""
            formatted_parts.append(
                f"[{p_label}]{section_info}: {passage.text.strip()}"
            )

        return "\n\n".join(formatted_parts), id_mapping

    def _format_training_text(
        self,
        question: str,
        passages_text: str,
        answer: str,
        cited_labels: list[str],
        answerability: AnswerabilityLabel,
        confidence: float,
    ) -> str:
        """Format a complete RAFT training example into text.

        Args:
            question: The clinical question.
            passages_text: Pre-formatted passages text with P-labels.
            answer: The target answer text.
            cited_labels: List of P-labels (e.g., ["P1", "P3"]) cited in answer.
            answerability: Answerability status.
            confidence: Confidence score for the answer.

        Returns:
            Formatted training text string.
        """
        citations_str = ", ".join(cited_labels) if cited_labels else "None"
        answerability_str = (
            answerability.value
            if isinstance(answerability, AnswerabilityLabel)
            else str(answerability)
        )

        user_content = (
            f"Question: {question.strip()}\n\n"
            f"Evidence:\n{passages_text}\n\n"
            "Instructions:\n"
            "1. Answer based ONLY on the provided evidence.\n"
            "2. Cite specific passage IDs for each claim.\n"
            "3. If evidence is insufficient, state INSUFFICIENT_EVIDENCE."
        )

        assistant_content = (
            f"ANSWER: {answer.strip()}\n"
            f"CITATIONS: {citations_str}\n"
            f"ANSWERABILITY: {answerability_str}\n"
            f"CONFIDENCE: {confidence:.2f}"
        )

        if self.config.template_style == "gemma":
            parts = [
                f"<start_of_turn>user\n{self.config.system_prompt}\n\n{user_content}<end_of_turn>",
                f"<start_of_turn>model\n{assistant_content}<end_of_turn>",
            ]
            return "\n".join(parts)

        # Default chat format
        formatted = f"System: {self.config.system_prompt}\n\n"
        formatted += f"User: {user_content}\n\n"
        formatted += f"Assistant: {assistant_content}"
        return formatted

    def build_answerable_example(
        self,
        question: str,
        gold_passages: list[Passage],
        answer: str,
        supporting_passage_ids: list[str],
    ) -> dict[str, Any]:
        """Build an answerable RAFT training example with gold + distractors.

        Args:
            question: Clinical question text.
            gold_passages: Passages that contain the answer.
            answer: Gold answer text.
            supporting_passage_ids: IDs of passages supporting the answer.

        Returns:
            Dictionary with formatted training text and metadata.
        """
        gold_ids = {p.passage_id for p in gold_passages}
        exclude_docs = {p.doc_id for p in gold_passages}
        distractors = self._select_distractors(
            gold_ids,
            self.config.num_distractor_passages,
            exclude_doc_ids=exclude_docs,
        )

        all_passages = list(gold_passages) + distractors
        passages_text, id_mapping = self._format_passages(
            all_passages, shuffle=self.config.shuffle_passages
        )

        # Map supporting passage IDs to their P-labels
        cited_labels = [
            id_mapping[pid] for pid in supporting_passage_ids
            if pid in id_mapping
        ]

        formatted_text = self._format_training_text(
            question=question,
            passages_text=passages_text,
            answer=answer,
            cited_labels=cited_labels,
            answerability=AnswerabilityLabel.ANSWERABLE,
            confidence=0.95,
        )

        self._category_counts["answerable"] += 1

        return {
            "text": formatted_text,
            "question": question,
            "answer": answer,
            "category": "answerable",
            "num_gold_passages": len(gold_passages),
            "num_distractor_passages": len(distractors),
            "cited_labels": cited_labels,
            "answerability": AnswerabilityLabel.ANSWERABLE.value,
        }

    def build_unanswerable_example(
        self,
        question: str,
        gold_passage_ids: Optional[set[str]] = None,
    ) -> dict[str, Any]:
        """Build an unanswerable RAFT example (distractor-only, no gold passage).

        Args:
            question: Clinical question text.
            gold_passage_ids: IDs of the actual gold passages (to exclude them).

        Returns:
            Dictionary with formatted training text and metadata.
        """
        gold_passage_ids = gold_passage_ids or set()
        num_passages = self.config.num_distractor_passages + 1  # fill gold slot too
        distractors = self._select_distractors(
            gold_passage_ids,
            num_passages,
        )

        passages_text, _ = self._format_passages(
            distractors, shuffle=self.config.shuffle_passages
        )

        formatted_text = self._format_training_text(
            question=question,
            passages_text=passages_text,
            answer="INSUFFICIENT_EVIDENCE. The provided passages do not contain "
                   "sufficient information to answer this question.",
            cited_labels=[],
            answerability=AnswerabilityLabel.UNANSWERABLE,
            confidence=0.90,
        )

        self._category_counts["unanswerable"] += 1

        return {
            "text": formatted_text,
            "question": question,
            "answer": "INSUFFICIENT_EVIDENCE",
            "category": "unanswerable",
            "num_gold_passages": 0,
            "num_distractor_passages": len(distractors),
            "cited_labels": [],
            "answerability": AnswerabilityLabel.UNANSWERABLE.value,
        }

    def build_from_raft_examples(
        self,
        examples: list[TrainingRAFTExample],
    ) -> list[dict[str, Any]]:
        """Build RAFT training dataset from TrainingRAFTExample instances.

        For each example, creates answerable and (optionally) unanswerable
        variants based on distractor_ratio.

        Args:
            examples: List of TrainingRAFTExample with gold and distractor passages.

        Returns:
            List of formatted training record dictionaries.
        """
        records: list[dict[str, Any]] = []
        self._category_counts = {
            "answerable": 0,
            "unanswerable": 0,
            "distractor_heavy": 0,
        }

        for ex in examples:
            # Always build the answerable version
            answerable_record = self.build_answerable_example(
                question=ex.question,
                gold_passages=ex.relevant_passages,
                answer=ex.answer,
                supporting_passage_ids=ex.supporting_passage_ids,
            )
            records.append(answerable_record)

            # Build unanswerable version based on distractor ratio
            if self.config.include_unanswerable:
                if self._rng.random() < self.config.distractor_ratio:
                    gold_ids = {p.passage_id for p in ex.relevant_passages}
                    unanswerable_record = self.build_unanswerable_example(
                        question=ex.question,
                        gold_passage_ids=gold_ids,
                    )
                    records.append(unanswerable_record)

        # Shuffle the final dataset
        self._rng.shuffle(records)

        logger.info(
            "Built RAFT dataset",
            total_examples=len(records),
            **self._category_counts,
        )
        return records

    def build_from_qa_pairs(
        self,
        questions: list[str],
        answers: list[str],
        gold_passages: list[list[Passage]],
        supporting_ids: Optional[list[list[str]]] = None,
    ) -> list[dict[str, Any]]:
        """Build RAFT dataset from parallel lists of questions, answers, and passages.

        Args:
            questions: List of clinical questions.
            answers: List of gold answers.
            gold_passages: List of lists of gold passages (one list per question).
            supporting_ids: Optional list of lists of supporting passage IDs.
                If None, all gold passage IDs are used.

        Returns:
            List of formatted training record dictionaries.

        Raises:
            ValueError: If input lists have mismatched lengths.
        """
        if not (len(questions) == len(answers) == len(gold_passages)):
            raise ValueError(
                f"Input length mismatch: questions={len(questions)}, "
                f"answers={len(answers)}, gold_passages={len(gold_passages)}"
            )

        records: list[dict[str, Any]] = []
        self._category_counts = {
            "answerable": 0,
            "unanswerable": 0,
            "distractor_heavy": 0,
        }

        for i, (question, answer, golds) in enumerate(
            zip(questions, answers, gold_passages)
        ):
            sup_ids = (
                supporting_ids[i] if supporting_ids and i < len(supporting_ids)
                else [p.passage_id for p in golds]
            )

            # Build answerable example
            answerable_record = self.build_answerable_example(
                question=question,
                gold_passages=golds,
                answer=answer,
                supporting_passage_ids=sup_ids,
            )
            records.append(answerable_record)

            # Build unanswerable example based on ratio
            if self.config.include_unanswerable:
                if self._rng.random() < self.config.distractor_ratio:
                    gold_ids = {p.passage_id for p in golds}
                    unanswerable_record = self.build_unanswerable_example(
                        question=question,
                        gold_passage_ids=gold_ids,
                    )
                    records.append(unanswerable_record)

        self._rng.shuffle(records)

        logger.info(
            "Built RAFT dataset from QA pairs",
            total_examples=len(records),
            **self._category_counts,
        )
        return records

    def to_hf_dataset(self, records: list[dict[str, Any]]) -> Any:
        """Convert list of record dicts to a HuggingFace Dataset.

        Args:
            records: List of training record dictionaries.

        Returns:
            HuggingFace Dataset or raw list if datasets is unavailable.
        """
        if not DATASETS_AVAILABLE or Dataset is None:
            logger.warning(
                "HF datasets not installed; returning raw list of records."
            )
            return records

        # Flatten cited_labels lists to comma-separated strings for HF Dataset
        serialized = []
        for rec in records:
            row = dict(rec)
            if isinstance(row.get("cited_labels"), list):
                row["cited_labels"] = ",".join(row["cited_labels"])
            serialized.append(row)

        return Dataset.from_list(serialized)

    def validate_dataset(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        """Validate RAFT dataset quality and composition.

        Args:
            records: List of training record dictionaries.

        Returns:
            Validation report dictionary.
        """
        total = len(records)
        category_dist: dict[str, int] = {}
        empty_texts = 0
        missing_answers = 0
        avg_passage_count = 0.0
        answerability_dist: dict[str, int] = {}

        for rec in records:
            cat = rec.get("category", "unknown")
            category_dist[cat] = category_dist.get(cat, 0) + 1

            text = rec.get("text", "")
            if not text or not str(text).strip():
                empty_texts += 1

            answer = rec.get("answer", "")
            if not answer or not str(answer).strip():
                missing_answers += 1

            n_gold = rec.get("num_gold_passages", 0)
            n_dist = rec.get("num_distractor_passages", 0)
            avg_passage_count += n_gold + n_dist

            ans_label = rec.get("answerability", "unknown")
            answerability_dist[ans_label] = answerability_dist.get(ans_label, 0) + 1

        avg_passage_count = (avg_passage_count / total) if total > 0 else 0.0

        answerable_count = category_dist.get("answerable", 0)
        unanswerable_count = category_dist.get("unanswerable", 0)
        actual_ratio = (
            unanswerable_count / (answerable_count + unanswerable_count)
            if (answerable_count + unanswerable_count) > 0
            else 0.0
        )

        report = {
            "total_samples": total,
            "category_distribution": category_dist,
            "answerability_distribution": answerability_dist,
            "empty_texts": empty_texts,
            "missing_answers": missing_answers,
            "avg_passages_per_example": round(avg_passage_count, 2),
            "actual_distractor_ratio": round(actual_ratio, 4),
            "configured_distractor_ratio": self.config.distractor_ratio,
            "is_valid": (
                empty_texts == 0
                and missing_answers == 0
                and total > 0
                and answerable_count > 0
            ),
        }

        logger.info("RAFT dataset validation completed", **report)
        return report

    def split_dataset(
        self,
        records: list[dict[str, Any]],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> dict[str, list[dict[str, Any]]]:
        """Split RAFT dataset into train/validation/test sets.

        Performs a stratified split to maintain the category distribution
        across splits.

        Args:
            records: List of training record dictionaries.
            train_ratio: Proportion for training set.
            val_ratio: Proportion for validation set.
            test_ratio: Proportion for test set.

        Returns:
            Dictionary with 'train', 'validation', 'test' splits.

        Raises:
            ValueError: If ratios don't sum to approximately 1.0.
        """
        total_ratio = train_ratio + val_ratio + test_ratio
        if not (0.99 <= total_ratio <= 1.01):
            raise ValueError(f"Split ratios must sum to 1.0, got: {total_ratio}")

        # Group by category for stratified splitting
        by_category: dict[str, list[dict[str, Any]]] = {}
        for rec in records:
            cat = rec.get("category", "other")
            by_category.setdefault(cat, []).append(rec)

        train_records: list[dict[str, Any]] = []
        val_records: list[dict[str, Any]] = []
        test_records: list[dict[str, Any]] = []

        for cat, cat_records in by_category.items():
            self._rng.shuffle(cat_records)
            n = len(cat_records)
            n_train = int(round(n * train_ratio))
            n_val = int(round(n * val_ratio))

            train_records.extend(cat_records[:n_train])
            val_records.extend(cat_records[n_train:n_train + n_val])
            test_records.extend(cat_records[n_train + n_val:])

        # Final shuffles
        self._rng.shuffle(train_records)
        self._rng.shuffle(val_records)
        self._rng.shuffle(test_records)

        logger.info(
            "RAFT dataset split complete",
            train=len(train_records),
            validation=len(val_records),
            test=len(test_records),
        )

        return {
            "train": train_records,
            "validation": val_records,
            "test": test_records,
        }
