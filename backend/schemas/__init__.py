"""API schemas package for MedFusion-RAFT."""

from backend.schemas.requests import (
    BatchQuestionRequest,
    EvaluationRequest,
    QuestionRequest,
)
from backend.schemas.responses import (
    AnswerResponse,
    BatchAnswerResponse,
    ClaimResponse,
    ErrorResponse,
    HealthResponse,
    PassageResponse,
)

__all__ = [
    "QuestionRequest",
    "BatchQuestionRequest",
    "EvaluationRequest",
    "PassageResponse",
    "ClaimResponse",
    "AnswerResponse",
    "BatchAnswerResponse",
    "HealthResponse",
    "ErrorResponse",
]
