"""End-to-end orchestrator for the MedFusion-RAFT pipeline.

Coordinates question routing, multi-stage retrieval, MedGemma clinical generation,
Qwen/NLI claim verification, and calibrated selective abstention into a unified QA system.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from src.models.base import (
    AnswerabilityStatus,
    BaseGeneratorProvider,
    BaseModelProvider,
    BaseVerifierProvider,
    GenerationOutput,
    ModelConfig,
)
from src.models.factory import create_generator, create_verifier
from src.models.schemas import (
    AnswerabilityLabel,
    DecisionType,
    ExperimentConfig,
    GeneratedAnswer,
    MedicalQuestion,
    PipelineDecision,
    PipelineResult,
    RetrievedPassage,
    VerificationResult,
)
from src.retrieval.pipeline import RetrievalPipeline, RetrievalPipelineConfig
from src.routing.router import RoutingDecision, RuleBasedRouter
from src.utils.config import get_app_config, load_app_config
from src.utils.logging import get_logger, log_execution_time

logger = get_logger(__name__)


class OrchestratorConfig(BaseModel):
    """Runtime orchestration parameters and feature flags."""

    model_config = ConfigDict(extra="ignore")

    enable_retrieval: bool = Field(default=True, description="Enable retrieval subsystem")
    enable_verification: bool = Field(default=True, description="Enable verification subsystem")
    enable_abstention: bool = Field(default=True, description="Enable calibrated abstention")
    enable_routing: bool = Field(default=False, description="Enable adaptive query routing")
    max_revision_attempts: int = Field(default=1, ge=0, description="Max iterative revision cycles")
    timeout_seconds: float = Field(default=120.0, gt=0.0, description="Per-query timeout ceiling")
    log_detailed_steps: bool = Field(default=True, description="Emit granular stage telemetry")
    experiment_id: Optional[str] = Field(default=None, description="Active experiment ID tag")


class MedFusionOrchestrator:
    """Master orchestrator executing the full MedFusion-RAFT inference lifecycle."""

    def __init__(
        self,
        generator: Optional[BaseGeneratorProvider] = None,
        retrieval_pipeline: Optional[RetrievalPipeline] = None,
        verification_pipeline: Optional[Any] = None,
        decision_engine: Optional[Any] = None,
        router: Optional[RuleBasedRouter] = None,
        config: Optional[OrchestratorConfig] = None,
    ) -> None:
        """Initialize the pipeline orchestrator with injected dependencies.

        Args:
            generator: Clinical answer generator (MedGemma or mock).
            retrieval_pipeline: Hybrid retrieval pipeline (BM25 + MedCPT).
            verification_pipeline: Multi-stage claim and citation verifier.
            decision_engine: Calibrated policy engine for accept/revise/abstain.
            router: Question complexity and routing classifier.
            config: Orchestrator configuration options.
        """
        self.generator = generator
        self.retrieval_pipeline = retrieval_pipeline
        self.verification_pipeline = verification_pipeline
        self.decision_engine = decision_engine
        self.router = router
        self.config = config or OrchestratorConfig()

        logger.info(
            "MedFusionOrchestrator initialized",
            retrieval=self.config.enable_retrieval and self.retrieval_pipeline is not None,
            verification=self.config.enable_verification and self.verification_pipeline is not None,
            abstention=self.config.enable_abstention and self.decision_engine is not None,
            routing=self.config.enable_routing and self.router is not None,
            max_revisions=self.config.max_revision_attempts,
            timeout_s=self.config.timeout_seconds,
        )

    def _format_context(self, passages: list[RetrievedPassage]) -> str:
        """Format retrieved passages into structured evidence text for prompting.

        Args:
            passages: Ordered list of retrieved passage objects.

        Returns:
            Concatenated formatted evidence string with bracketed identifiers.
        """
        if not passages:
            return ""

        formatted_blocks: list[str] = []
        for i, p in enumerate(passages):
            pid = p.passage_id if p.passage_id else f"P{i+1}"
            doc_id = p.doc_id if p.doc_id else "unknown_source"
            block = f"[{pid}] (Source: {doc_id})\n{p.text.strip()}"
            formatted_blocks.append(block)

        return "\n\n".join(formatted_blocks)

    def _parse_generator_output(
        self, gen_output: GenerationOutput, question: MedicalQuestion
    ) -> GeneratedAnswer:
        """Transform raw generator GenerationOutput into structured GeneratedAnswer.

        Args:
            gen_output: Raw generation output container.
            question: Original medical question.

        Returns:
            Validated GeneratedAnswer instance.
        """
        parsed_dict = gen_output.metadata.get("parsed_answer", {})
        raw_text = gen_output.text.strip()

        answer_text = parsed_dict.get("answer") or raw_text
        # Clean answer text if it includes raw ANSWER: headers
        if answer_text.startswith("ANSWER:"):
            answer_text = answer_text.replace("ANSWER:", "", 1).strip()
            # If multi-line, take only up to CITATIONS or other keys
            for marker in ["\nCITATIONS:", "\nANSWERABILITY:", "\nCONFIDENCE:"]:
                if marker in answer_text:
                    answer_text = answer_text.split(marker, 1)[0].strip()

        raw_ans = str(parsed_dict.get("answerability", "answerable")).lower()
        if "unanswerable" in raw_ans:
            ans_label = AnswerabilityLabel.UNANSWERABLE
        elif "partially" in raw_ans:
            ans_label = AnswerabilityLabel.PARTIALLY_ANSWERABLE
        else:
            ans_label = AnswerabilityLabel.ANSWERABLE

        confidence = parsed_dict.get("confidence")
        if confidence is None:
            confidence = 0.9 if ans_label == AnswerabilityLabel.ANSWERABLE else 0.3

        model_id = gen_output.metadata.get("model_id") or getattr(
            self.generator, "config", ModelConfig(model_id="generator")
        ).model_id

        return GeneratedAnswer(
            answer_text=answer_text,
            claims=[],
            answerability=ans_label,
            confidence=float(confidence),
            reasoning=gen_output.metadata.get("mode"),
            generation_time_ms=gen_output.generation_time_ms,
            model_id=str(model_id),
        )

    def _handle_revision(
        self,
        question: MedicalQuestion,
        answer: GeneratedAnswer,
        verification: VerificationResult,
        passages: list[RetrievedPassage],
    ) -> GeneratedAnswer:
        """Regenerate clinical answer incorporating verifier feedback and critique.

        Args:
            question: Original medical inquiry.
            answer: Flawed candidate answer.
            verification: Verification analysis with critique and ungrounded claims.
            passages: Retrieved evidence passages.

        Returns:
            Revised GeneratedAnswer instance.
        """
        context = self._format_context(passages)
        critique = verification.verification_reasoning or "Some claims were unverified or weak."
        revision_prompt = (
            f"CLINICAL QUESTION: {question.question}\n\n"
            f"INITIAL ANSWER: {answer.answer_text}\n\n"
            f"CRITIQUE & EVIDENCE GAPS: {critique}\n\n"
            f"EVIDENCE PASSAGES:\n{context}\n\n"
            "INSTRUCTIONS: Revise the answer strictly grounded in the evidence passages. "
            "Remove unsupported claims and cite valid passage IDs explicitly."
        )

        logger.info(
            "Executing answer revision",
            question_id=question.question_id,
            unsupported=verification.unsupported_claims,
            contradicted=verification.contradicted_claims,
        )

        gen_kwargs = {"temperature": 0.05, "max_new_tokens": 512}
        if self.generator is None:
            return answer

        revised_output = self.generator.generate(revision_prompt, **gen_kwargs)
        revised_answer = self._parse_generator_output(revised_output, question)
        revised_answer.reasoning = f"Revised after critique: {critique[:120]}"
        return revised_answer

    def _build_final_response(
        self,
        question: MedicalQuestion,
        answer: Optional[GeneratedAnswer],
        verification: Optional[VerificationResult],
        decision: Optional[PipelineDecision],
        passages: list[RetrievedPassage],
        total_time_ms: float,
    ) -> PipelineResult:
        """Construct the end-to-end PipelineResult container.

        Args:
            question: Input question.
            answer: Candidate generated answer.
            verification: Verification assessment.
            decision: Final policy disposition.
            passages: Context passages utilized.
            total_time_ms: Overall pipeline latency in ms.

        Returns:
            Complete PipelineResult.
        """
        final_text: Optional[str] = None
        if decision is not None and decision.decision == DecisionType.ABSTAIN:
            reason = decision.reasoning if decision.reasoning else "Insufficient evidence"
            final_text = (
                "MedFusion Abstention: I cannot provide a clinically reliable answer for this "
                f"inquiry based on the retrieved evidence. (Disposition reason: {reason})"
            )
        elif answer is not None:
            final_text = answer.answer_text
        else:
            final_text = "MedFusion Abstention: Answer generation failed."

        return PipelineResult(
            question=question,
            retrieved_passages=passages,
            generated_answer=answer,
            verification_result=verification,
            decision=decision,
            final_answer=final_text,
            total_time_ms=total_time_ms,
            experiment_id=self.config.experiment_id,
        )

    def process_question(self, question: MedicalQuestion) -> PipelineResult:
        """Execute the complete MedFusion-RAFT inference pipeline for a single question.

        Stages:
            1. Input validation
            2. Question classification / routing
            3. Multi-modality hybrid retrieval
            4. Clinical generation with evidence context
            5. Independent verification & citation validation
            6. Policy decision (accept / revise / abstain)
            7. Iterative revision if required
            8. Final response compilation

        Args:
            question: MedicalQuestion specification.

        Returns:
            PipelineResult containing answer, verification, and telemetry.
        """
        t0 = time.perf_counter()

        # Stage 1: Input Validation
        if not question or not question.question or not question.question.strip():
            logger.warning("Empty or invalid question received", question_id=getattr(question, "question_id", "none"))
            decision = PipelineDecision(
                decision=DecisionType.ABSTAIN,
                score=0.0,
                reasoning="Empty or invalid question text provided",
            )
            return self._build_final_response(question, None, None, decision, [], 0.0)

        logger.info(
            "Processing clinical question",
            question_id=question.question_id,
            text_snippet=question.question[:80],
        )

        # Stage 2: Question Classification / Routing
        routing_decision = RoutingDecision.STANDARD
        retrieval_params: dict[str, Any] = {}
        gen_params: dict[str, Any] = {}
        ver_params: dict[str, Any] = {}

        if self.config.enable_routing and self.router is not None:
            with log_execution_time(logger, "question_routing"):
                routing_decision = self.router.classify_question(question.question)
                retrieval_params = self.router.get_retrieval_config(routing_decision)
                gen_params = self.router.get_generation_config(routing_decision)
                ver_params = self.router.get_verification_config(routing_decision)

        # Check timeout early
        if (time.perf_counter() - t0) > self.config.timeout_seconds:
            dec = PipelineDecision(
                decision=DecisionType.ABSTAIN, score=0.0, reasoning="Pipeline timeout during routing"
            )
            return self._build_final_response(question, None, None, dec, [], (time.perf_counter() - t0) * 1000)

        # Stage 3: Retrieval
        passages: list[RetrievedPassage] = []
        if self.config.enable_retrieval and self.retrieval_pipeline is not None:
            try:
                with log_execution_time(logger, "hybrid_retrieval"):
                    top_k = retrieval_params.get("final_top_k")
                    passages = self.retrieval_pipeline.retrieve(question.question, top_k=top_k)
            except Exception as exc:
                logger.error("Retrieval failed: %s", exc, exc_info=True)
                passages = []

        context_str = self._format_context(passages)

        # Stage 4: Generation
        answer: Optional[GeneratedAnswer] = None
        if self.generator is not None:
            try:
                with log_execution_time(logger, "answer_generation"):
                    gen_output = self.generator.generate_answer(
                        question=question.question,
                        context=context_str,
                        **gen_params,
                    )
                    answer = self._parse_generator_output(gen_output, question)
            except Exception as exc:
                logger.error("Generation failed: %s", exc, exc_info=True)
                dec = PipelineDecision(
                    decision=DecisionType.ABSTAIN,
                    score=0.0,
                    reasoning=f"Generator inference failed: {exc}",
                )
                return self._build_final_response(
                    question, None, None, dec, passages, (time.perf_counter() - t0) * 1000
                )
        else:
            dec = PipelineDecision(
                decision=DecisionType.ABSTAIN, score=0.0, reasoning="No generator configured"
            )
            return self._build_final_response(
                question, None, None, dec, passages, (time.perf_counter() - t0) * 1000
            )

        # Stage 5: Verification
        verification: Optional[VerificationResult] = None
        verification_failed: bool = False
        if self.config.enable_verification and self.verification_pipeline is not None:
            try:
                with log_execution_time(logger, "verification"):
                    verification = self.verification_pipeline.verify(
                        question=question.question,
                        answer=answer,
                        passages=passages,
                    )
            except Exception as exc:
                logger.error("Verification failed: %s", exc, exc_info=True)
                # CRITICAL FAIL-SAFE: Verification failure MUST result in ABSTAIN, not ACCEPT
                verification_failed = True
                verification = None

        # Stage 6: Decision Policy
        decision: Optional[PipelineDecision] = None
        
        # FAIL-SAFE: If verification was enabled but failed, we MUST abstain
        if verification_failed:
            decision = PipelineDecision(
                decision=DecisionType.ABSTAIN,
                score=0.0,
                reasoning="Verification system failed - cannot validate answer safety",
                revision_count=0,
            )
        elif self.config.enable_abstention and self.decision_engine is not None and verification is not None:
            decision = self.decision_engine.decide(
                answer=answer,
                verification=verification,
                num_evidence_passages=len(passages),
                current_revision_count=0,
            )
        else:
            # Default decision policy when abstention or verification is intentionally bypassed in config
            conf = answer.confidence if (answer and answer.confidence is not None) else 1.0
            decision = PipelineDecision(
                decision=DecisionType.ACCEPT,
                score=conf,
                reasoning="Bypassed abstention engine (verification or abstention disabled in experiment config)",
                revision_count=0,
            )

        # Stage 7: Iterative Revision
        revision_attempts = 0
        while (
            decision.decision == DecisionType.REVISE
            and revision_attempts < self.config.max_revision_attempts
            and (time.perf_counter() - t0) < self.config.timeout_seconds
        ):
            revision_attempts += 1
            logger.info("Starting revision iteration %d", revision_attempts)
            assert verification is not None  # Required for revision feedback

            answer = self._handle_revision(question, answer, verification, passages)

            # Re-verify revised answer
            if self.config.enable_verification and self.verification_pipeline is not None:
                try:
                    verification = self.verification_pipeline.verify(
                        question=question.question,
                        answer=answer,
                        passages=passages,
                    )
                except Exception as exc:
                    logger.error("Re-verification error during revision: %s", exc)
                    # FAIL-SAFE: If re-verification fails, we must abstain
                    decision = PipelineDecision(
                        decision=DecisionType.ABSTAIN,
                        score=0.0,
                        reasoning=f"Re-verification failed during revision attempt {revision_attempts}",
                        revision_count=revision_attempts,
                    )
                    break  # Exit revision loop

            if self.decision_engine is not None and verification is not None:
                decision = self.decision_engine.decide(
                    answer=answer,
                    verification=verification,
                    num_evidence_passages=len(passages),
                    current_revision_count=revision_attempts,
                )
            else:
                decision = PipelineDecision(
                    decision=DecisionType.ACCEPT,
                    score=answer.confidence or 0.8,
                    reasoning=f"Accepted after revision {revision_attempts}",
                    revision_count=revision_attempts,
                )

        total_time_ms = (time.perf_counter() - t0) * 1000.0
        return self._build_final_response(
            question=question,
            answer=answer,
            verification=verification,
            decision=decision,
            passages=passages,
            total_time_ms=total_time_ms,
        )

    def process_batch(self, questions: list[MedicalQuestion]) -> list[PipelineResult]:
        """Process a sequence of medical questions sequentially.

        Args:
            questions: List of MedicalQuestion records.

        Returns:
            List of PipelineResult outcomes.
        """
        logger.info("Processing batch of %d questions", len(questions))
        results: list[PipelineResult] = []
        for q in questions:
            res = self.process_question(q)
            results.append(res)
        return results


def create_orchestrator_from_config(
    config_dir: Union[str, Path] = "./configs",
    experiment_id: str = "E1_base",
    use_mock_models: bool = False,
) -> MedFusionOrchestrator:
    """Factory creating fully assembled MedFusionOrchestrator based on experiment specs.

    Args:
        config_dir: Path to directory containing YAML configs.
        experiment_id: Experiment identifier (e.g., 'E1_base', 'E3_hybrid_rag', 'E9_full_system').
        use_mock_models: If True, forces MockGenerator and MockVerifier for CPU/testing.

    Returns:
        Configured and wired MedFusionOrchestrator instance.
    """
    app_cfg = load_app_config(config_dir)
    exp_spec: Optional[ExperimentConfig] = app_cfg.experiments.get(experiment_id)

    if exp_spec is None:
        logger.warning(
            "Experiment '%s' not found in config; falling back to default base run", experiment_id
        )
        exp_spec = ExperimentConfig(
            experiment_id=experiment_id,
            name=f"Fallback {experiment_id}",
            generator="base",
            retrieval=None,
            fine_tuning=None,
            verification=False,
            abstention=False,
            routing=False,
        )

    # 1. Generator Provider
    gen_provider_type = "mock" if use_mock_models else app_cfg.models.generator.model_type
    
    # Build complete generator config with all YAML fields
    gen_config_dict = {
        "model_id": app_cfg.models.generator.model_id,
        "provider_type": gen_provider_type,
        "revision": app_cfg.models.generator.revision,
        "device_map": app_cfg.models.generator.device_map,
        "trust_remote_code": app_cfg.models.generator.trust_remote_code,
        "max_new_tokens": app_cfg.models.generator.generation.max_new_tokens,
        "temperature": app_cfg.models.generator.generation.temperature,
        "top_p": app_cfg.models.generator.generation.top_p,
        "do_sample": app_cfg.models.generator.generation.do_sample,
        "repetition_penalty": app_cfg.models.generator.generation.repetition_penalty,
    }
    
    # Add quantization config if present and enabled
    if hasattr(app_cfg.models.generator, 'quantization') and app_cfg.models.generator.quantization:
        quant = app_cfg.models.generator.quantization
        if hasattr(quant, 'enabled') and quant.enabled:
            gen_config_dict.update({
                "quantization_bits": quant.bits if hasattr(quant, 'bits') else 4,
                "quant_type": quant.quant_type if hasattr(quant, 'quant_type') else "nf4",
                "compute_dtype": quant.compute_dtype if hasattr(quant, 'compute_dtype') else "bfloat16",
                "use_double_quant": quant.use_double_quant if hasattr(quant, 'use_double_quant') else True,
            })
    
    generator = create_generator(config=ModelConfig(**gen_config_dict))

    # 2. Retrieval Pipeline
    retrieval_pipeline: Optional[RetrievalPipeline] = None
    if exp_spec.retrieval is not None:
        is_hybrid = exp_spec.retrieval == "hybrid"
        is_dense = exp_spec.retrieval == "dense" or is_hybrid
        is_bm25 = exp_spec.retrieval == "bm25" or is_hybrid
        ret_cfg = RetrievalPipelineConfig(
            enable_bm25=is_bm25,
            enable_dense=is_dense,
            enable_fusion=is_hybrid,
            enable_reranker=is_hybrid,
            use_mock_components=use_mock_models,
        )
        retrieval_pipeline = RetrievalPipeline(config=ret_cfg)

    # 3. Verification Pipeline
    verification_pipeline = None
    if exp_spec.verification:
        from src.verification.pipeline import VerificationPipeline, VerificationPipelineConfig
        from src.verification.entailment import EntailmentChecker, EntailmentConfig, MockEntailmentChecker

        ver_provider_type = "mock" if use_mock_models else app_cfg.models.verifier.model_type
        
        # Build complete verifier config with all YAML fields
        ver_config_dict = {
            "model_id": app_cfg.models.verifier.model_id,
            "provider_type": ver_provider_type,
            "revision": app_cfg.models.verifier.revision,
            "device_map": app_cfg.models.verifier.device_map,
            "trust_remote_code": app_cfg.models.verifier.trust_remote_code,
            "max_new_tokens": app_cfg.models.verifier.generation.max_new_tokens,
            "temperature": app_cfg.models.verifier.generation.temperature,
            "top_p": app_cfg.models.verifier.generation.top_p,
        }
        
        # Add quantization config if present and enabled
        if hasattr(app_cfg.models.verifier, 'quantization') and app_cfg.models.verifier.quantization:
            quant = app_cfg.models.verifier.quantization
            if hasattr(quant, 'enabled') and quant.enabled:
                ver_config_dict.update({
                    "quantization_bits": quant.bits if hasattr(quant, 'bits') else 4,
                    "quant_type": quant.quant_type if hasattr(quant, 'quant_type') else "nf4",
                    "compute_dtype": quant.compute_dtype if hasattr(quant, 'compute_dtype') else "bfloat16",
                    "use_double_quant": quant.use_double_quant if hasattr(quant, 'use_double_quant') else True,
                })
        
        verifier = create_verifier(config=ModelConfig(**ver_config_dict))
        
        # Initialize entailment checker: use real NLI model unless mock mode
        entailment_checker = None
        if use_mock_models:
            entailment_checker = MockEntailmentChecker()
            logger.info("Using MockEntailmentChecker for testing (use_mock_models=True)")
        else:
            # Use real NLI model for production
            entailment_config = EntailmentConfig(
                model_id="cross-encoder/nli-deberta-v3-base",
                threshold_supported=0.7,
                threshold_contradicted=0.3,
                device=app_cfg.models.verifier.device_map if hasattr(app_cfg.models.verifier, 'device_map') else "cpu",
            )
            entailment_checker = EntailmentChecker(config=entailment_config)
            logger.info("Using real EntailmentChecker with NLI model: %s", entailment_config.model_id)
        
        ver_cfg = VerificationPipelineConfig()
        verification_pipeline = VerificationPipeline(
            config=ver_cfg,
            verifier_provider=verifier,
            entailment_checker=entailment_checker,
        )

    # 4. Decision Engine
    decision_engine = None
    if exp_spec.abstention:
        from src.verification.decision import DecisionEngine, DecisionPolicyConfig

        dec_cfg = DecisionPolicyConfig()
        decision_engine = DecisionEngine(config=dec_cfg)

    # 5. Router
    router = None
    if exp_spec.routing:
        router = RuleBasedRouter()

    # 6. Orchestrator Config
    orch_config = OrchestratorConfig(
        enable_retrieval=exp_spec.retrieval is not None,
        enable_verification=exp_spec.verification,
        enable_abstention=exp_spec.abstention,
        enable_routing=exp_spec.routing,
        experiment_id=exp_spec.experiment_id,
    )

    return MedFusionOrchestrator(
        generator=generator,
        retrieval_pipeline=retrieval_pipeline,
        verification_pipeline=verification_pipeline,
        decision_engine=decision_engine,
        router=router,
        config=orch_config,
    )
