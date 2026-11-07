"""Response data models for MedFusion-RAFT API."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class PassageResponse(BaseModel):
    """Retrieved or cited evidence passage chunk response representation."""

    model_config = ConfigDict(extra="allow")

    passage_id: str = Field(..., description="Unique passage chunk identifier.")
    text: str = Field(..., description="Text content of the retrieved passage chunk.")
    source: Optional[str] = Field(default=None, description="Document source or filename.")
    score: float = Field(default=0.0, description="Relevance or retrieval fusion rank score.")
    section: Optional[str] = Field(default=None, description="Document section header.")


class ClaimResponse(BaseModel):
    """Atomic claim extracted from candidate answer with verification outcome."""

    model_config = ConfigDict(extra="allow")

    claim_text: str = Field(..., description="Extracted atomic claim proposition.")
    support_status: str = Field(
        default="unverified",
        description="Grounding status (supported, partially_supported, unsupported, contradicted, insufficient_evidence).",
    )
    evidence_passage_id: Optional[str] = Field(
        default=None,
        description="ID of supporting evidence passage chunk, if linked.",
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Detailed entailment or critique rationale from verifier.",
    )


class AnswerResponse(BaseModel):
    """End-to-end question answering pipeline response."""

    model_config = ConfigDict(extra="allow")

    answer_text: str = Field(..., description="User-facing generated answer or abstention message.")
    citations: list[str] = Field(
        default_factory=list,
        description="Passage identifiers cited within the answer text.",
    )
    answerability: str = Field(
        default="answerable",
        description="Predicted answerability (answerable, partially_answerable, unanswerable).",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for release decision or answer quality.",
    )
    claims: list[ClaimResponse] = Field(
        default_factory=list,
        description="List of decomposed atomic claims and their verification states.",
    )
    verification_status: str = Field(
        default="unverified",
        description="High-level verification state (verified, unverified, contradicted).",
    )
    decision: str = Field(
        default="accept",
        description="Pipeline policy decision (accept, revise, abstain).",
    )
    evidence_passages: list[PassageResponse] = Field(
        default_factory=list,
        description="Retrieved evidence passages providing grounding context.",
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Total round-trip processing latency in milliseconds.",
    )
    experiment_id: Optional[str] = Field(
        default=None,
        description="Experiment configuration identifier used during processing.",
    )


class BatchAnswerResponse(BaseModel):
    """Batch answer payload returned for BatchQuestionRequest."""

    model_config = ConfigDict(extra="allow")

    results: list[AnswerResponse] = Field(
        ...,
        description="List of answers corresponding to each input question.",
    )
    total_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Total batch processing latency in milliseconds.",
    )
    experiment_id: Optional[str] = Field(
        default=None,
        description="Active experiment identifier for the batch run.",
    )


class HealthResponse(BaseModel):
    """System health check and diagnostic status response."""

    model_config = ConfigDict(extra="allow")

    status: str = Field(default="healthy", description="Overall service operational health.")
    models_loaded: bool = Field(
        default=False,
        description="Indicates whether ML model checkpoints are active in memory.",
    )
    gpu_available: bool = Field(
        default=False,
        description="Indicates whether CUDA acceleration is available on host.",
    )
    uptime_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Total backend service uptime in seconds.",
    )
    version: str = Field(
        default="0.1.0",
        description="Service software release version.",
    )


class ErrorResponse(BaseModel):
    """Standardized API error envelope."""

    model_config = ConfigDict(extra="allow")

    error: str = Field(..., description="High-level error classification or summary.")
    detail: Optional[str] = Field(default=None, description="Detailed error trace or message.")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when error occurred.",
    )
