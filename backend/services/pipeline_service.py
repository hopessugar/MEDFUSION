"""Pipeline orchestration service powering the MedFusion-RAFT API.

This service wraps the main MedFusionOrchestrator and provides API-friendly
interfaces, eliminating duplicate pipeline implementations.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Sequence

from backend.schemas.requests import BatchQuestionRequest, QuestionRequest
from backend.schemas.responses import (
    AnswerResponse,
    BatchAnswerResponse,
    ClaimResponse,
    HealthResponse,
    PassageResponse,
)
from src.models.schemas import (
    AnswerabilityLabel,
    DecisionType,
    MedicalQuestion,
    PipelineResult,
)
from src.pipeline.orchestrator import (
    MedFusionOrchestrator,
    create_orchestrator_from_config,
)
from src.utils.config import AppConfig, get_app_config

logger = logging.getLogger(__name__)


class PipelineService:
    """Service encapsulating the main MedFusionOrchestrator for API use.
    
    This is a thin wrapper that translates API requests to/from the canonical
    orchestrator, ensuring a single source of truth for pipeline execution.
    """

    _instance: Optional[PipelineService] = None

    def __init__(self, config_dir: str = "./configs") -> None:
        """Initialize pipeline service with configuration directory.

        Args:
            config_dir: Directory containing application YAML configs.
        """
        self.config_dir = config_dir
        self.config: AppConfig = get_app_config(config_dir=config_dir)
        self.orchestrator: Optional[MedFusionOrchestrator] = None

        self._start_time: float = time.time()
        self._models_loaded: bool = False
        self._mock_mode: bool = False
        self._current_experiment_id: str = "E1_base"

    @classmethod
    def get_instance(cls, config_dir: str = "./configs", mock: bool = False) -> PipelineService:
        """Retrieve or create singleton instance of PipelineService.

        Args:
            config_dir: Configuration directory path.
            mock: Whether to initialize in mock mode.

        Returns:
            Configured singleton PipelineService.
        """
        if cls._instance is None:
            cls._instance = cls(config_dir=config_dir)
            cls._instance.initialize(mock=mock)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (useful for clean tear downs in testing)."""
        if cls._instance is not None:
            cls._instance.shutdown()
            cls._instance = None

    def initialize(self, mock: bool = False, experiment_id: str = "E1_base") -> None:
        """Instantiate the main orchestrator with selected experiment configuration.

        Args:
            mock: If True, uses mock providers for rapid CPU-friendly execution.
            experiment_id: Experiment configuration to load.
        """
        if self._models_loaded:
            logger.info("PipelineService already initialized.")
            return

        self._mock_mode = mock
        self._current_experiment_id = experiment_id
        logger.info(
            "Initializing PipelineService (mock=%s, experiment=%s)...",
            mock,
            experiment_id,
        )

        # Create the main orchestrator using factory
        try:
            self.orchestrator = create_orchestrator_from_config(
                config_dir=self.config_dir,
                experiment_id=experiment_id,
                use_mock_models=mock,
            )
            self._models_loaded = True
            logger.info("PipelineService initialization complete.")
        except Exception as exc:
            logger.error("Failed to initialize orchestrator: %s", exc, exc_info=True)
            # Fall back to mock mode if real models fail
            if not mock:
                logger.warning("Falling back to mock mode due to initialization failure")
                self._mock_mode = True
                self.orchestrator = create_orchestrator_from_config(
                    config_dir=self.config_dir,
                    experiment_id=experiment_id,
                    use_mock_models=True,
                )
                self._models_loaded = True
            else:
                raise

    def _convert_request_to_question(self, request: QuestionRequest) -> MedicalQuestion:
        """Convert API request to internal MedicalQuestion schema."""
        return MedicalQuestion(
            question_id=f"api_{int(time.time() * 1000000)}",
            question=request.question,
            specialty=request.specialty,
            metadata={"source": "api", "timestamp": datetime.now(timezone.utc).isoformat()},
        )

    def _convert_result_to_response(self, result: PipelineResult) -> AnswerResponse:
        """Convert internal PipelineResult to API response schema."""
        # Extract claims
        claim_responses: list[ClaimResponse] = []
        if result.verification_result and result.verification_result.claim_results:
            for claim in result.verification_result.claim_results:
                claim_responses.append(
                    ClaimResponse(
                        claim_text=claim.claim_text,
                        support_status=claim.support_status.value if claim.support_status else "unknown",
                        evidence_passage_id=(
                            claim.citations[0].passage_id if claim.citations else None
                        ),
                        reasoning=None,
                    )
                )

        # Extract passages
        evidence_responses = [
            PassageResponse(
                passage_id=p.passage_id,
                text=p.text,
                source=p.metadata.get("source") if p.metadata else None,
                score=float(p.score),
                section=p.section,
            )
            for p in result.retrieved_passages
        ]

        # Extract citations
        citations = []
        if result.generated_answer and result.generated_answer.claims:
            for claim in result.generated_answer.claims:
                for cit in claim.citations:
                    if cit.passage_id not in citations:
                        citations.append(cit.passage_id)

        return AnswerResponse(
            answer_text=result.final_answer or "Unable to generate answer",
            citations=citations,
            answerability=(
                result.generated_answer.answerability.value
                if result.generated_answer
                else AnswerabilityLabel.UNANSWERABLE.value
            ),
            confidence=(
                result.generated_answer.confidence if result.generated_answer else 0.0
            ),
            claims=claim_responses,
            verification_status=(
                "verified"
                if result.verification_result and result.verification_result.verified
                else "not_verified"
            ),
            decision=(
                result.decision.decision.value if result.decision else DecisionType.ACCEPT.value
            ),
            evidence_passages=evidence_responses,
            processing_time_ms=round(result.total_time_ms, 2),
            experiment_id=result.experiment_id,
        )

    def process_question(self, request: QuestionRequest) -> AnswerResponse:
        """Process a single clinical question end-to-end using main orchestrator.

        Args:
            request: Validated QuestionRequest payload.

        Returns:
            Structured AnswerResponse with grounded answer, claims, and decision.
        """
        if not self._models_loaded:
            self.initialize(mock=self._mock_mode, experiment_id=request.experiment_id or self._current_experiment_id)

        # Reinitialize if experiment changed (different config)
        if request.experiment_id and request.experiment_id != self._current_experiment_id:
            logger.info("Switching experiment from %s to %s", self._current_experiment_id, request.experiment_id)
            self._current_experiment_id = request.experiment_id
            # Note: In production, you might want to cache multiple orchestrators
            # For now, we reinitialize
            self.orchestrator = create_orchestrator_from_config(
                config_dir=self.config_dir,
                experiment_id=request.experiment_id,
                use_mock_models=self._mock_mode,
            )

        question = self._convert_request_to_question(request)
        
        assert self.orchestrator is not None, "Orchestrator not initialized"
        result = self.orchestrator.process_question(question)
        
        return self._convert_result_to_response(result)

    def process_batch(self, request: BatchQuestionRequest) -> BatchAnswerResponse:
        """Process a batch of questions using main orchestrator.

        Args:
            request: Validated BatchQuestionRequest payload.

        Returns:
            BatchAnswerResponse containing all generated answers.
        """
        start_time = time.perf_counter()
        results: list[AnswerResponse] = []

        for q_req in request.questions:
            # Propagate batch experiment_id if not explicitly set on item
            if request.experiment_id and not q_req.experiment_id:
                q_req.experiment_id = request.experiment_id
            ans = self.process_question(q_req)
            results.append(ans)

        total_elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return BatchAnswerResponse(
            results=results,
            total_time_ms=round(total_elapsed_ms, 2),
            experiment_id=request.experiment_id,
        )

    def get_health(self) -> HealthResponse:
        """Query service status and hardware availability.

        Returns:
            Diagnostic HealthResponse instance.
        """
        gpu_available = False
        try:
            import torch

            gpu_available = torch.cuda.is_available()
        except ImportError:
            pass

        uptime = time.time() - self._start_time

        # Check if models are actually loaded (not just initialized)
        models_loaded = self._models_loaded and self.orchestrator is not None
        if models_loaded and self.orchestrator:
            # Check generator status
            if self.orchestrator.generator:
                models_loaded = models_loaded and self.orchestrator.generator.is_loaded

        return HealthResponse(
            status="healthy" if models_loaded else "initializing",
            models_loaded=models_loaded,
            gpu_available=gpu_available,
            uptime_seconds=round(uptime, 2),
            version="0.2.0",  # Incremented after orchestrator unification
        )

    def shutdown(self) -> None:
        """Unload models and release allocated resources."""
        logger.info("Shutting down PipelineService...")
        
        # The orchestrator's models will be garbage collected
        # Individual providers handle their own cleanup in __del__ if needed
        self.orchestrator = None
        self._models_loaded = False
        
        logger.info("PipelineService successfully shut down.")
