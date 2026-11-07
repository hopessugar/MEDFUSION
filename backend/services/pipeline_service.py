"""Pipeline orchestration service powering the MedFusion-RAFT API."""

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
from src.models.base import (
    BaseGeneratorProvider,
    BaseVerifierProvider,
    ParsedMedicalAnswer,
    ParsedVerificationResult,
)
from src.models.factory import create_generator, create_verifier
from src.models.medgemma import MedGemmaProvider
from src.models.qwen_verifier import QwenVerifierProvider
from src.models.schemas import (
    AnswerabilityLabel,
    AtomicClaim,
    Citation,
    ClaimSupportStatus,
    DecisionType,
    Document,
    GeneratedAnswer,
    Passage,
    PipelineDecision,
    RetrievedPassage,
    VerificationResult,
)
from src.retrieval.pipeline import RetrievalPipeline, RetrievalPipelineConfig
from src.utils.config import AppConfig, get_app_config
from src.verification.citation_validator import CitationValidator
from src.verification.claim_extractor import ClaimExtractor
from src.verification.decision import DecisionEngine, DecisionPolicyConfig

logger = logging.getLogger(__name__)

# Fallback default clinical guidelines indexed for immediate out-of-the-box grounding
DEFAULT_CLINICAL_CORPUS: list[Passage] = [
    Passage(
        passage_id="guideline_htn_001",
        doc_id="acc_aha_htn_2024",
        text=(
            "First-line pharmacological agents for uncomplicated primary hypertension include "
            "thiazide diuretics, calcium channel blockers (CCBs), and angiotensin-converting enzyme "
            "(ACE) inhibitors or angiotensin receptor blockers (ARBs). Beta-blockers are not "
            "recommended as initial monotherapy in the absence of compelling indications such as "
            "ischemic heart disease or heart failure."
        ),
        section="Pharmacological Management",
        metadata={"source": "ACC/AHA Clinical Guidelines", "title": "Hypertension Guideline"},
    ),
    Passage(
        passage_id="guideline_t2d_002",
        doc_id="ada_diabetes_2024",
        text=(
            "Metformin remains the cornerstone first-line pharmacotherapy for type 2 diabetes "
            "mellitus in patients without severe renal impairment (eGFR < 30 mL/min/1.73m²). "
            "For patients with established atherosclerotic cardiovascular disease, heart failure, "
            "or chronic kidney disease, SGLT2 inhibitors or GLP-1 receptor agonists with proven "
            "cardiovascular and renal benefit should be incorporated."
        ),
        section="Glycemic Control",
        metadata={"source": "ADA Standards of Care", "title": "Diabetes Care Standards"},
    ),
    Passage(
        passage_id="guideline_asthma_003",
        doc_id="gina_asthma_2024",
        text=(
            "Global Initiative for Asthma (GINA) guidelines no longer recommend SABA-only treatment "
            "of asthma. For adults and adolescents, the preferred Track 1 reliever is low-dose "
            "inhaled corticosteroid (ICS)-formoterol as needed, which significantly reduces the "
            "risk of severe exacerbations compared to short-acting beta2-agonist monotherapy."
        ),
        section="Asthma Controller Therapy",
        metadata={"source": "GINA Report", "title": "Global Strategy for Asthma Management"},
    ),
]


class PipelineService:
    """Service encapsulating retrieval, generation, verification, and abstention routing."""

    _instance: Optional[PipelineService] = None

    def __init__(self, config_dir: str = "./configs") -> None:
        """Initialize pipeline service with configuration directory.

        Args:
            config_dir: Directory containing application YAML configs.
        """
        self.config_dir = config_dir
        self.config: AppConfig = get_app_config(config_dir=config_dir)
        self.generator: Optional[BaseGeneratorProvider] = None
        self.verifier: Optional[BaseVerifierProvider] = None
        self.retrieval_pipeline: Optional[RetrievalPipeline] = None
        self.claim_extractor: Optional[ClaimExtractor] = None
        self.citation_validator: Optional[CitationValidator] = None
        self.decision_engine: Optional[DecisionEngine] = None

        self._start_time: float = time.time()
        self._models_loaded: bool = False
        self._mock_mode: bool = False

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

    def initialize(self, mock: bool = False) -> None:
        """Instantiate models, retrieval index, and verification engines.

        Args:
            mock: If True, uses mock providers for rapid CPU-friendly execution.
        """
        if self._models_loaded:
            logger.info("PipelineService already initialized.")
            return

        self._mock_mode = mock
        logger.info("Initializing PipelineService (mock=%s)...", mock)

        provider_type = "mock" if mock else None

        # 1. Generator
        try:
            self.generator = create_generator(provider_type=provider_type)
            self.generator.load()
        except Exception as exc:
            logger.warning("Failed to initialize primary generator, falling back to mock: %s", exc)
            self.generator = create_generator(provider_type="mock")
            self.generator.load()

        # 2. Verifier
        try:
            self.verifier = create_verifier(provider_type=provider_type)
            self.verifier.load()
        except Exception as exc:
            logger.warning("Failed to initialize primary verifier, falling back to mock: %s", exc)
            self.verifier = create_verifier(provider_type="mock")
            self.verifier.load()

        # 3. Retrieval Pipeline
        retrieval_cfg = RetrievalPipelineConfig(
            use_mock_components=mock or not os.environ.get("CUDA_VISIBLE_DEVICES"),
            bm25_top_k=5,
            dense_top_k=5,
            final_top_k=3,
        )
        self.retrieval_pipeline = RetrievalPipeline(config=retrieval_cfg)
        self.retrieval_pipeline.build_index(DEFAULT_CLINICAL_CORPUS)

        # 4. Verification components
        self.claim_extractor = ClaimExtractor()
        self.citation_validator = CitationValidator()
        self.decision_engine = DecisionEngine(config=DecisionPolicyConfig())

        self._models_loaded = True
        logger.info("PipelineService initialization complete.")

    def process_question(self, request: QuestionRequest) -> AnswerResponse:
        """Process a single clinical question end-to-end.

        Args:
            request: Validated QuestionRequest payload.

        Returns:
            Structured AnswerResponse with grounded answer, claims, and decision.
        """
        if not self._models_loaded:
            self.initialize(mock=self._mock_mode)

        start_time = time.perf_counter()
        retrieved_passages: list[RetrievedPassage] = []

        # 1. Retrieval Stage
        if request.use_retrieval and self.retrieval_pipeline is not None:
            try:
                retrieved_passages = self.retrieval_pipeline.retrieve(request.question, top_k=3)
            except Exception as exc:
                logger.error("Error during retrieval: %s", exc, exc_info=True)
                retrieved_passages = []

        context_texts = [p.text for p in retrieved_passages]
        evidence_responses = [
            PassageResponse(
                passage_id=p.passage_id,
                text=p.text,
                source=p.metadata.get("source") if p.metadata else None,
                score=float(p.score),
                section=p.section,
            )
            for p in retrieved_passages
        ]

        # 2. Generation Stage
        assert self.generator is not None
        gen_output = self.generator.generate_answer(
            question=request.question,
            context=context_texts,
        )
        parsed_answer = MedGemmaProvider.parse_medical_answer(gen_output.text)

        answer_text = parsed_answer.answer
        citations = list(parsed_answer.citations)
        confidence = float(parsed_answer.confidence)
        answerability = parsed_answer.answerability.value

        # 3. Verification Stage
        claim_responses: list[ClaimResponse] = []
        verification_status = "unverified"
        decision_label = "accept"
        contradicted_count = 0
        supported_count = 0

        if request.use_verification and self.verifier is not None:
            ver_output = self.verifier.verify_answer(
                question=request.question,
                answer=answer_text,
                evidence=context_texts,
            )
            parsed_ver = QwenVerifierProvider.parse_verification_output(ver_output.text)
            verification_status = parsed_ver.overall_assessment.value

            for c in parsed_ver.claims:
                claim_responses.append(
                    ClaimResponse(
                        claim_text=c.claim,
                        support_status=c.status.value,
                        evidence_passage_id=c.evidence if c.evidence != "NONE" else None,
                        reasoning=c.reasoning,
                    )
                )
                if c.status.value == "contradicted":
                    contradicted_count += 1
                elif c.status.value == "supported":
                    supported_count += 1

            # 4. Decision Policy
            if request.use_abstention and self.decision_engine is not None:
                if answerability == AnswerabilityLabel.UNANSWERABLE.value:
                    decision_label = DecisionType.ABSTAIN.value
                    answer_text = (
                        "The system cannot answer this question based on currently retrieved evidence."
                    )
                elif contradicted_count > 0:
                    decision_label = DecisionType.ABSTAIN.value
                    answer_text = (
                        "Clinical assertion withheld due to contradictory evidence in published guidelines."
                    )
                elif verification_status in ("not_verified", "insufficient_evidence") and supported_count == 0:
                    decision_label = DecisionType.REVISE.value
                else:
                    decision_label = DecisionType.ACCEPT.value
        elif not request.use_verification:
            verification_status = "skipped"

        total_elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return AnswerResponse(
            answer_text=answer_text,
            citations=citations,
            answerability=answerability,
            confidence=confidence,
            claims=claim_responses,
            verification_status=verification_status,
            decision=decision_label,
            evidence_passages=evidence_responses,
            processing_time_ms=round(total_elapsed_ms, 2),
            experiment_id=request.experiment_id,
        )

    def process_batch(self, request: BatchQuestionRequest) -> BatchAnswerResponse:
        """Process a batch of questions sequentially or concurrently.

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

        return HealthResponse(
            status="healthy",
            models_loaded=self._models_loaded,
            gpu_available=gpu_available,
            uptime_seconds=round(uptime, 2),
            version="0.1.0",
        )

    def shutdown(self) -> None:
        """Unload models and release allocated resources."""
        logger.info("Shutting down PipelineService...")
        if self.generator is not None:
            try:
                self.generator.unload()
            except Exception as exc:
                logger.warning("Error unloading generator: %s", exc)
            self.generator = None

        if self.verifier is not None:
            try:
                self.verifier.unload()
            except Exception as exc:
                logger.warning("Error unloading verifier: %s", exc)
            self.verifier = None

        self._models_loaded = False
        logger.info("PipelineService successfully shut down.")
