"""Request data models for MedFusion-RAFT API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class QuestionRequest(BaseModel):
    """Payload for submitting a single clinical question."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        ...,
        min_length=3,
        description="Clinical query or question text presented for grounding.",
        examples=["What is the recommended first-line treatment for uncomplicated hypertension?"],
    )
    specialty: Optional[str] = Field(
        default=None,
        description="Target clinical specialty (e.g., Cardiology, Endocrinology).",
        examples=["Cardiology"],
    )
    use_retrieval: bool = Field(
        default=True,
        description="Whether to execute sparse/dense retrieval of evidence passages.",
    )
    use_verification: bool = Field(
        default=True,
        description="Whether to perform claim extraction and factual verification.",
    )
    use_abstention: bool = Field(
        default=True,
        description="Whether to apply selective abstention policies on ungrounded answers.",
    )
    experiment_id: Optional[str] = Field(
        default=None,
        description="Experiment configuration identifier (e.g., E1_base, E5_raft).",
        examples=["E5_raft"],
    )


class BatchQuestionRequest(BaseModel):
    """Payload for processing multiple questions concurrently or sequentially."""

    model_config = ConfigDict(extra="forbid")

    questions: list[QuestionRequest] = Field(
        ...,
        min_length=1,
        description="Batch collection of questions to process through the pipeline.",
    )
    experiment_id: Optional[str] = Field(
        default=None,
        description="Global experiment override applied to all questions in batch if not set per item.",
    )


class EvaluationRequest(BaseModel):
    """Payload for executing an automated benchmark or dataset evaluation."""

    model_config = ConfigDict(extra="forbid")

    dataset_path: str = Field(
        ...,
        description="Filesystem path to the benchmark dataset file (.json, .jsonl, .csv).",
        examples=["data/processed/pubmedqa_test.json"],
    )
    experiment_id: str = Field(
        ...,
        description="Experiment identifier configuring generator, retrieval, and verification modules.",
        examples=["E5_raft"],
    )
    output_dir: Optional[str] = Field(
        default=None,
        description="Optional target directory path for saving evaluation output artifacts.",
    )
