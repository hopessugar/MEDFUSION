"""FastAPI API routes for MedFusion-RAFT."""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from backend.schemas.requests import (
    BatchQuestionRequest,
    EvaluationRequest,
    QuestionRequest,
)
from backend.schemas.responses import (
    AnswerResponse,
    BatchAnswerResponse,
    ErrorResponse,
    HealthResponse,
)
from backend.services.pipeline_service import PipelineService
from src.utils.config import get_app_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Medical QA Pipeline"])


# Security: Allowed base directories for file operations
ALLOWED_DATA_DIRS = {
    Path("./data").resolve(),
    Path("./experiments").resolve(),
}


def validate_file_path(file_path: str, operation: str = "read") -> Path:
    """Validate file path to prevent path traversal attacks.
    
    Args:
        file_path: User-provided file path string.
        operation: Type of operation ('read' or 'write').
    
    Returns:
        Validated and resolved Path object.
    
    Raises:
        HTTPException: If path is invalid or outside allowed directories.
    """
    try:
        # Resolve to absolute path
        resolved_path = Path(file_path).resolve()
        
        # Check if path is within allowed directories
        is_allowed = any(
            resolved_path.is_relative_to(allowed_dir) or resolved_path == allowed_dir
            for allowed_dir in ALLOWED_DATA_DIRS
        )
        
        if not is_allowed:
            logger.warning(
                "Path traversal attempt blocked: %s (resolved: %s)",
                file_path,
                resolved_path,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Access denied: path must be within allowed directories "
                    f"(data/, experiments/). Attempted path: {file_path}"
                ),
            )
        
        # For read operations, verify file exists
        if operation == "read" and not resolved_path.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File not found: {file_path}",
            )
        
        # For write operations, ensure parent directory exists
        if operation == "write":
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
        
        return resolved_path
        
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Path validation error for %s: %s", file_path, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file path: {str(exc)}",
        ) from exc


def get_pipeline_service() -> PipelineService:
    """Dependency injection provider for the pipeline singleton service."""
    return PipelineService.get_instance()


@router.post(
    "/ask",
    response_model=AnswerResponse,
    status_code=status.HTTP_200_OK,
    summary="Answer a clinical question",
    description="Processes a medical question through retrieval, generation, verification, and abstention routing.",
)
async def ask_question(
    request: QuestionRequest,
    service: PipelineService = Depends(get_pipeline_service),
) -> AnswerResponse:
    """Execute end-to-end question answering pipeline for a single clinical query."""
    try:
        return service.process_question(request)
    except Exception as exc:
        logger.error("Failed processing question: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline inference error: {str(exc)}",
        ) from exc


@router.post(
    "/batch",
    response_model=BatchAnswerResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch answer clinical questions",
    description="Processes an array of clinical questions with shared or individual configurations.",
)
async def ask_batch(
    request: BatchQuestionRequest,
    service: PipelineService = Depends(get_pipeline_service),
) -> BatchAnswerResponse:
    """Execute end-to-end pipeline over a batch of questions."""
    try:
        return service.process_batch(request)
    except Exception as exc:
        logger.error("Failed processing batch: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch execution error: {str(exc)}",
        ) from exc


@router.post(
    "/evaluate",
    status_code=status.HTTP_200_OK,
    summary="Run benchmark dataset evaluation",
    description="Executes a benchmark evaluation over a specified dataset file using an experiment config.",
)
async def run_evaluation(
    request: EvaluationRequest,
    service: PipelineService = Depends(get_pipeline_service),
) -> Dict[str, Any]:
    """Execute evaluation over questions in a benchmark dataset file."""
    # Validate dataset path for security
    dataset_path = validate_file_path(request.dataset_path, operation="read")

    # Load dataset records
    records: list[dict[str, Any]] = []
    suffix = dataset_path.suffix.lower()

    try:
        if suffix == ".json":
            with open(dataset_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                records = loaded if isinstance(loaded, list) else [loaded]
        elif suffix == ".jsonl":
            with open(dataset_path, "r", encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]
        elif suffix in (".csv", ".tsv"):
            delim = "\t" if suffix == ".tsv" else ","
            with open(dataset_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter=delim)
                records = list(reader)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported evaluation dataset format: {suffix}",
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error reading dataset file: {str(exc)}",
        ) from exc

    if not records:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset contains no sample records.",
        )

    # Process evaluation batch
    question_requests: list[QuestionRequest] = []
    for rec in records:
        q_text = rec.get("question") or rec.get("query") or rec.get("prompt")
        if q_text and len(str(q_text).strip()) >= 3:
            question_requests.append(
                QuestionRequest(
                    question=str(q_text).strip(),
                    specialty=rec.get("specialty"),
                    experiment_id=request.experiment_id,
                )
            )

    batch_req = BatchQuestionRequest(
        questions=question_requests,
        experiment_id=request.experiment_id,
    )
    batch_resp = service.process_batch(batch_req)

    # Validate and determine output location
    out_dir_path = request.output_dir or "./experiments/evaluations"
    out_dir = validate_file_path(out_dir_path, operation="write").parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / out_dir_path.split("/")[-1] if "/" in out_dir_path else out_dir / f"eval_{request.experiment_id}_{int(time.time())}.json"

    eval_result = {
        "experiment_id": request.experiment_id,
        "dataset_path": str(dataset_path),
        "num_evaluated": len(batch_resp.results),
        "total_time_ms": batch_resp.total_time_ms,
        "avg_time_per_question_ms": (
            batch_resp.total_time_ms / len(batch_resp.results) if batch_resp.results else 0.0
        ),
        "decisions": {
            "accept": sum(1 for r in batch_resp.results if r.decision == "accept"),
            "revise": sum(1 for r in batch_resp.results if r.decision == "revise"),
            "abstain": sum(1 for r in batch_resp.results if r.decision == "abstain"),
        },
        "output_file": str(out_file),
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": eval_result,
                "answers": [a.model_dump() for a in batch_resp.results],
            },
            f,
            indent=2,
        )

    return eval_result


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Service health check",
    description="Returns service availability, model loading state, and hardware capabilities.",
)
async def check_health(
    service: PipelineService = Depends(get_pipeline_service),
) -> HealthResponse:
    """Return runtime diagnostic and health status."""
    return service.get_health()


@router.get(
    "/config",
    status_code=status.HTTP_200_OK,
    summary="Retrieve active configuration",
    description="Returns active model, retrieval, and experiment configuration parameters.",
)
async def get_configuration() -> Dict[str, Any]:
    """Return active application configuration dictionary."""
    app_cfg = get_app_config()
    return app_cfg.model_dump()
