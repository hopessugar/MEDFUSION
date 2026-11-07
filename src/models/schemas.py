"""Core data models and schemas for MedFusion-RAFT.

This module defines all Pydantic models and enumerations representing documents,
passages, questions, claims, verification states, training instances, and evaluation
metrics across the MedFusion-RAFT pipeline.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AnswerabilityLabel(str, Enum):
    """Classification of whether a clinical question can be answered from evidence."""

    ANSWERABLE = "answerable"
    PARTIALLY_ANSWERABLE = "partially_answerable"
    UNANSWERABLE = "unanswerable"


class ClaimSupportStatus(str, Enum):
    """Verification status for an atomic clinical claim against retrieved evidence."""

    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class DecisionType(str, Enum):
    """Pipeline policy decision regarding answer release or refinement."""

    ACCEPT = "accept"
    REVISE = "revise"
    ABSTAIN = "abstain"


class QuestionDifficulty(str, Enum):
    """Assigned difficulty level for medical QA benchmarks."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class DocumentMetadata(BaseModel):
    """Metadata describing a source clinical document or guideline."""

    model_config = ConfigDict(extra="allow", frozen=True)

    doc_id: str = Field(..., description="Unique document identifier")
    title: str = Field(..., description="Title of the source document or paper")
    source: str = Field(..., description="Source repository, journal, or guideline corpus")
    url: Optional[str] = Field(default=None, description="Direct URL to source if available")
    publication_date: Optional[str] = Field(default=None, description="Publication date string")
    license: Optional[str] = Field(default=None, description="Licensing terms of the document")
    version: Optional[str] = Field(default=None, description="Corpus or document release version")


class Document(BaseModel):
    """Source clinical document or guideline representation."""

    model_config = ConfigDict(extra="allow")

    doc_id: str = Field(..., description="Unique document identifier")
    title: str = Field(default="", description="Title of the source document or paper")
    text: str = Field(..., min_length=1, description="Full text content of the document")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Metadata dictionary"
    )
    pages: Optional[list[str]] = Field(
        default=None, description="Optional page-by-page text if available"
    )


class Passage(BaseModel):
    """A discrete textual passage chunk extracted from a clinical document."""

    model_config = ConfigDict(extra="allow")

    passage_id: str = Field(..., description="Unique identifier for this text passage")
    doc_id: str = Field(..., description="Reference ID to parent document")
    text: str = Field(..., min_length=1, description="Passage content text")
    section: Optional[str] = Field(default=None, description="Section heading from source")
    page: Optional[int] = Field(default=None, ge=1, description="Page number in source document")
    chunk_index: int = Field(default=0, ge=0, description="Sequential chunk index in document")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Associated document metadata"
    )

    @field_validator("metadata", mode="before")
    @classmethod
    def _validate_metadata(cls, v: Any) -> dict[str, Any]:
        if v is None:
            return {}
        if isinstance(v, BaseModel):
            return v.model_dump()
        if isinstance(v, dict):
            return v
        return dict(v)


class RetrievedPassage(Passage):
    """Passage retrieved via sparse, dense, or hybrid retrieval methods."""

    score: float = Field(default=0.0, description="Relevance score from retrieval or reranking")
    retrieval_method: str = Field(
        default="unknown", description="Method used (bm25, dense, rrf, cross_encoder)"
    )
    rank: int = Field(default=0, ge=0, description="1-based or 0-based rank in retrieval results")


class MedicalQuestion(BaseModel):
    """Input clinical question presented to the MedFusion system."""

    model_config = ConfigDict(extra="allow")

    question_id: str = Field(..., description="Unique identifier for the question")
    question: str = Field(..., min_length=3, description="Clinical question text")
    specialty: Optional[str] = Field(
        default=None, description="Clinical specialty (e.g., Cardiology, Oncology)"
    )
    difficulty: Optional[QuestionDifficulty] = Field(
        default=None, description="Assessed question difficulty"
    )
    expected_answerability: Optional[AnswerabilityLabel] = Field(
        default=None, description="Ground truth answerability label if known"
    )


class Citation(BaseModel):
    """Citation linking a claim directly to a supporting retrieved passage."""

    claim_text: str = Field(..., description="Text of the cited claim")
    passage_id: str = Field(..., description="ID of the passage providing evidence")
    relevance_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Attribution or relevance score"
    )


class AtomicClaim(BaseModel):
    """An atomic, verifiable proposition extracted from a generated answer."""

    claim_id: str = Field(..., description="Unique claim identifier")
    claim_text: str = Field(..., min_length=3, description="Verifiable proposition statement")
    citations: list[Citation] = Field(
        default_factory=list, description="Passages cited for this claim"
    )
    support_status: Optional[ClaimSupportStatus] = Field(
        default=None, description="Verification outcome"
    )
    entailment_score: Optional[float] = Field(
        default=None, description="NLI entailment confidence score"
    )


class GeneratedAnswer(BaseModel):
    """Candidate answer produced by the generator model."""

    answer_text: str = Field(..., description="Full text generated by the clinical model")
    claims: list[AtomicClaim] = Field(
        default_factory=list, description="Decomposed atomic propositions"
    )
    answerability: AnswerabilityLabel = Field(
        default=AnswerabilityLabel.ANSWERABLE,
        description="Predicted answerability of the question",
    )
    confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Generator self-confidence"
    )
    reasoning: Optional[str] = Field(
        default=None, description="Chain-of-thought or rationale trace"
    )
    generation_time_ms: Optional[float] = Field(
        default=None, ge=0.0, description="Inference latency in milliseconds"
    )
    model_id: str = Field(default="", description="Identifier of the generating model checkpoint")


class VerificationResult(BaseModel):
    """Output from the Qwen / NLI verification subsystem."""

    verified: bool = Field(..., description="Whether answer satisfies grounding criteria")
    overall_score: float = Field(
        default=0.0, description="Aggregated verification score across claims"
    )
    claim_results: list[AtomicClaim] = Field(
        default_factory=list, description="Verification results per claim"
    )
    supported_claims: int = Field(default=0, ge=0, description="Count of supported claims")
    unsupported_claims: int = Field(default=0, ge=0, description="Count of unsupported claims")
    contradicted_claims: int = Field(default=0, ge=0, description="Count of contradicted claims")
    verification_reasoning: Optional[str] = Field(
        default=None, description="Verifier explanation or critique"
    )
    verification_time_ms: Optional[float] = Field(
        default=None, ge=0.0, description="Verification latency in milliseconds"
    )
    verifier_model_id: str = Field(default="", description="Model used for verification")


class PipelineDecision(BaseModel):
    """Routing or policy decision for answer disposition."""

    decision: DecisionType = Field(..., description="Selected action: accept, revise, or abstain")
    score: float = Field(..., description="Decision confidence or composite safety score")
    reasoning: str = Field(..., description="Explicit rationale for disposition")
    revision_count: int = Field(
        default=0, ge=0, description="Number of iterative revisions executed"
    )


class PipelineResult(BaseModel):
    """End-to-end execution outcome of the MedFusion-RAFT pipeline."""

    question: MedicalQuestion = Field(..., description="Input medical question")
    retrieved_passages: list[RetrievedPassage] = Field(
        default_factory=list, description="Retrieved passages used as context"
    )
    generated_answer: Optional[GeneratedAnswer] = Field(
        default=None, description="Candidate generated answer"
    )
    verification_result: Optional[VerificationResult] = Field(
        default=None, description="Verification evaluation result"
    )
    decision: Optional[PipelineDecision] = Field(
        default=None, description="Final disposition decision"
    )
    final_answer: Optional[str] = Field(
        default=None, description="Final user-facing answer or abstention response"
    )
    total_time_ms: float = Field(
        default=0.0, ge=0.0, description="Total pipeline latency in milliseconds"
    )
    experiment_id: Optional[str] = Field(
        default=None, description="Experiment configuration identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of execution",
    )


class DatasetExample(BaseModel):
    """Standardized medical QA benchmark or fine-tuning dataset record."""

    question_id: str = Field(..., description="Dataset unique sample identifier")
    question: str = Field(..., description="Medical query or clinical problem")
    reference_answer: str = Field(..., description="Gold reference standard answer")
    gold_passage: Optional[str] = Field(
        default=None, description="Gold passage text supporting answer"
    )
    gold_passage_id: Optional[str] = Field(default=None, description="ID of gold evidence passage")
    doc_id: Optional[str] = Field(default=None, description="Source document identifier")
    page: Optional[int] = Field(default=None, ge=1, description="Source document page")
    specialty: Optional[str] = Field(default=None, description="Medical specialty categorization")
    difficulty: Optional[QuestionDifficulty] = Field(
        default=None, description="Assigned difficulty"
    )
    answerability: AnswerabilityLabel = Field(
        default=AnswerabilityLabel.ANSWERABLE, description="Ground truth answerability"
    )
    source_url: Optional[str] = Field(default=None, description="Source URL or DOI")
    review_status: str = Field(default="unreviewed", description="Clinical expert review state")


class TrainingSFTExample(BaseModel):
    """Formatted instance for Supervised Fine-Tuning (SFT)."""

    question: str = Field(..., description="Prompt question")
    context: str = Field(..., description="Concatenated reference context passages")
    answer: str = Field(..., description="Target clinical response with citations")
    citation_ids: list[str] = Field(
        default_factory=list, description="List of passage IDs cited in answer"
    )
    answerability: AnswerabilityLabel = Field(
        default=AnswerabilityLabel.ANSWERABLE, description="Answerability status"
    )


class TrainingRAFTExample(BaseModel):
    """Formatted instance for Retrieval-Augmented Fine-Tuning (RAFT)."""

    question: str = Field(..., description="Clinical inquiry")
    relevant_passages: list[Passage] = Field(
        ..., description="Passages containing actual supporting facts"
    )
    distractor_passages: list[Passage] = Field(
        default_factory=list, description="Irrelevant distractor passages"
    )
    answer: str = Field(..., description="Chain-of-thought answer citing relevant passages")
    supporting_passage_ids: list[str] = Field(
        default_factory=list, description="IDs of ground-truth supporting passages"
    )
    answerability: AnswerabilityLabel = Field(
        default=AnswerabilityLabel.ANSWERABLE, description="Answerability status"
    )


class EvaluationMetrics(BaseModel):
    """Comprehensive evaluation metrics covering accuracy, retrieval, grounding, and risk."""

    accuracy: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    f1: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    exact_match: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    recall_at_k: Optional[dict[int, float]] = Field(default=None)
    precision_at_k: Optional[dict[int, float]] = Field(default=None)
    mrr: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    ndcg_at_k: Optional[dict[int, float]] = Field(default=None)
    claim_support_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    citation_precision: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    unsupported_claim_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    contradiction_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    abstention_precision: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    abstention_recall: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    coverage: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    selective_risk: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    ece: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Expected Calibration Error")
    brier_score: Optional[float] = Field(default=None, ge=0.0, description="Brier score")
    median_latency_ms: Optional[float] = Field(default=None, ge=0.0)
    p95_latency_ms: Optional[float] = Field(default=None, ge=0.0)
    peak_gpu_memory_mb: Optional[float] = Field(default=None, ge=0.0)


class ExperimentConfig(BaseModel):
    """Specification for an experimental ablation or benchmark run."""

    model_config = ConfigDict(extra="allow")

    experiment_id: str = Field(..., description="Unique experiment code (e.g., E1_base, E5_raft)")
    name: str = Field(..., description="Human-readable experiment label")
    generator: str = Field(..., description="Generator type: base, sft, raft, sft_raft")
    retrieval: Optional[str] = Field(
        default=None, description="Retrieval strategy: None, bm25, dense, hybrid"
    )
    fine_tuning: Optional[str] = Field(
        default=None, description="Fine-tuning regime: None, sft, raft, sft_raft"
    )
    verification: bool = Field(default=False, description="Whether verifier module is active")
    abstention: bool = Field(default=False, description="Whether selective abstention is active")
    routing: bool = Field(default=False, description="Whether adaptive query routing is active")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    notes: Optional[str] = Field(default=None, description="Optional experiment notes")


class ExperimentResult(BaseModel):
    """Collected results and metadata from an experiment execution."""

    config: ExperimentConfig = Field(..., description="Run configuration")
    metrics: EvaluationMetrics = Field(
        default_factory=EvaluationMetrics, description="Computed evaluation metrics"
    )
    num_examples: int = Field(default=0, ge=0, description="Number of evaluated samples")
    status: str = Field(
        default="NOT_RUN", description="Execution status: NOT_RUN, RUNNING, COMPLETED, FAILED"
    )
    started_at: Optional[datetime] = Field(default=None, description="Experiment start time")
    completed_at: Optional[datetime] = Field(default=None, description="Experiment completion time")
    hardware_info: Optional[dict[str, Any]] = Field(
        default=None, description="GPU/CPU environment information"
    )
