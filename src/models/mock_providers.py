"""Mock model providers for testing, development, and CPU-only environments."""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Optional, Union

from src.models.base import (
    AnswerabilityStatus,
    BaseGeneratorProvider,
    BaseVerifierProvider,
    ClaimVerificationStatus,
    GenerationOutput,
    ModelConfig,
    OverallAssessmentStatus,
    ParsedMedicalAnswer,
    ParsedVerificationResult,
    VerificationClaim,
)

logger = logging.getLogger(__name__)


class MockGeneratorProvider(BaseGeneratorProvider):
    """Deterministic mock generator for clinical QA testing without GPU dependencies."""

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        mode: str = "realistic",
        custom_answer: Optional[str] = None,
        simulated_latency_ms: float = 10.0,
    ) -> None:
        """Initialize mock generator.

        Args:
            config: Optional ModelConfig (defaults to mock config).
            mode: Behavior mode: 'realistic', 'always_accept', 'insufficient', 'contradiction', 'custom'.
            custom_answer: Pre-set answer string when mode is 'custom'.
            simulated_latency_ms: Artificial latency for timing simulations.
        """
        cfg = config or ModelConfig(
            model_id="mock-medical-generator",
            provider_type="mock",
        )
        super().__init__(cfg)
        self.mode = mode
        self.custom_answer = custom_answer
        self.simulated_latency_ms = simulated_latency_ms

    def load(self) -> None:
        """Simulate loading model weights."""
        self._loaded = True
        logger.info("MockGeneratorProvider loaded in mode '%s'", self.mode)

    def unload(self) -> None:
        """Simulate unloading model weights."""
        self._loaded = False
        logger.info("MockGeneratorProvider unloaded.")

    def get_memory_usage(self) -> dict[str, float]:
        """Return simulated memory statistics."""
        return {
            "gpu_allocated_mb": 0.0,
            "gpu_reserved_mb": 0.0,
            "system_ram_mb": 35.5,
        }

    def generate(self, prompt: str, **kwargs: Any) -> GenerationOutput:
        """Generate simulated text output."""
        if not self.is_loaded:
            self.load()

        input_tokens = max(1, len(prompt.split()))
        output_text = "MOCK GENERATION: Response grounded in simulated medical knowledge."
        output_tokens = len(output_text.split())

        return GenerationOutput(
            text=output_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            generation_time_ms=self.simulated_latency_ms,
            metadata={"mode": self.mode, "provider": "mock_generator"},
        )

    def generate_answer(
        self,
        question: str,
        context: Union[str, list[str]],
        **kwargs: Any,
    ) -> GenerationOutput:
        """Generate structured medical answer according to configured mode.

        Args:
            question: Medical query.
            context: Retrieved evidence context or list of passage strings.
            **kwargs: Extra parameters.

        Returns:
            GenerationOutput with formatted answer and parsed structure.
        """
        if not self.is_loaded:
            self.load()

        start_time = time.perf_counter()

        # Normalize context passages
        passages: list[str] = []
        if isinstance(context, str):
            passages = [p.strip() for p in context.split("\n\n") if p.strip()]
        else:
            passages = list(context)

        # Detect passage IDs
        found_pids = re.findall(r"\[P\d+\]", " ".join(passages))
        if not found_pids:
            found_pids = [f"[P{i+1}]" for i in range(min(len(passages), 3))]
        if not found_pids:
            found_pids = ["[P1]"]

        mode = kwargs.get("mode", self.mode)

        if mode == "custom" and self.custom_answer:
            answer_text = self.custom_answer
            citations = found_pids
            answerability = AnswerabilityStatus.ANSWERABLE
            confidence = 0.95

        elif mode == "insufficient" or not passages:
            answer_text = f"INSUFFICIENT_EVIDENCE. The provided passages do not contain adequate clinical evidence regarding: '{question}'."
            citations = []
            answerability = AnswerabilityStatus.UNANSWERABLE
            confidence = 0.10

        elif mode == "contradiction":
            answer_text = f"The evidence presents conflicting clinical recommendations regarding '{question}'. Passage {found_pids[0]} suggests positive outcomes, whereas other evidence contradicts this finding."
            citations = found_pids[:2]
            answerability = AnswerabilityStatus.PARTIALLY_ANSWERABLE
            confidence = 0.65

        else:  # 'realistic' or 'always_accept'
            summary_snippet = passages[0][:120].strip() if passages else "standard clinical protocols apply"
            if summary_snippet.startswith("[P"):
                summary_snippet = summary_snippet.split("]", 1)[-1].strip()
            answer_text = f"Based on evidence {', '.join(found_pids)}, {summary_snippet}. The therapy is indicated for patient management under structured medical supervision."
            citations = found_pids
            answerability = AnswerabilityStatus.ANSWERABLE
            confidence = 0.95

        formatted_output = (
            f"ANSWER: {answer_text}\n"
            f"CITATIONS: {', '.join(citations)}\n"
            f"ANSWERABILITY: {answerability.value}\n"
            f"CONFIDENCE: {confidence:.2f}"
        )

        input_tokens = len(question.split()) + sum(len(p.split()) for p in passages)
        output_tokens = len(formatted_output.split())
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0 + self.simulated_latency_ms

        parsed = ParsedMedicalAnswer(
            answer=answer_text,
            citations=citations,
            answerability=answerability,
            confidence=confidence,
            raw_text=formatted_output,
        )

        return GenerationOutput(
            text=formatted_output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            generation_time_ms=elapsed_ms,
            metadata={
                "provider": "mock_generator",
                "mode": mode,
                "parsed_answer": parsed.model_dump(),
            },
        )


class MockVerifierProvider(BaseVerifierProvider):
    """Deterministic mock verifier for fact-checking pipeline tests."""

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        mode: str = "realistic",
        simulated_latency_ms: float = 12.0,
    ) -> None:
        """Initialize mock verifier.

        Args:
            config: Optional ModelConfig.
            mode: Behavior mode: 'realistic', 'always_accept', 'always_reject', 'partially_verified', 'contradicted'.
            simulated_latency_ms: Artificial latency for timing simulation.
        """
        cfg = config or ModelConfig(
            model_id="mock-verifier-model",
            provider_type="mock",
        )
        super().__init__(cfg)
        self.mode = mode
        self.simulated_latency_ms = simulated_latency_ms

    def load(self) -> None:
        """Simulate loading verifier."""
        self._loaded = True
        logger.info("MockVerifierProvider loaded in mode '%s'", self.mode)

    def unload(self) -> None:
        """Simulate unloading verifier."""
        self._loaded = False
        logger.info("MockVerifierProvider unloaded.")

    def get_memory_usage(self) -> dict[str, float]:
        """Return simulated memory statistics."""
        return {
            "gpu_allocated_mb": 0.0,
            "gpu_reserved_mb": 0.0,
            "system_ram_mb": 42.0,
        }

    def generate(self, prompt: str, **kwargs: Any) -> GenerationOutput:
        """Generate mock raw verification text."""
        if not self.is_loaded:
            self.load()

        input_tokens = len(prompt.split())
        output_text = "OVERALL_ASSESSMENT: VERIFIED\nSUPPORT_SCORE: 1.0\nISSUES: None"
        return GenerationOutput(
            text=output_text,
            input_tokens=input_tokens,
            output_tokens=len(output_text.split()),
            generation_time_ms=self.simulated_latency_ms,
            metadata={"mode": self.mode, "provider": "mock_verifier"},
        )

    def verify_answer(
        self,
        question: str,
        answer: str,
        evidence: Union[str, list[str]],
        **kwargs: Any,
    ) -> GenerationOutput:
        """Execute mock verification on given answer against evidence.

        Args:
            question: Original question.
            answer: Answer generated by candidate model.
            evidence: Evidence passages list or context string.
            **kwargs: Extra parameters.

        Returns:
            GenerationOutput with structured claims, status, and scores.
        """
        if not self.is_loaded:
            self.load()

        start_time = time.perf_counter()
        mode = kwargs.get("mode", self.mode)

        passages: list[str] = []
        if isinstance(evidence, str):
            passages = [p.strip() for p in evidence.split("\n\n") if p.strip()]
        else:
            passages = list(evidence)

        # Detect passage IDs
        found_pids = re.findall(r"\[P\d+\]", " ".join(passages))
        primary_pid = found_pids[0] if found_pids else "[P1]"

        # Break answer into sentences for mock claims
        clean_ans = re.sub(r"^(ANSWER:|CITATIONS:|CONFIDENCE:).*", "", answer, flags=re.MULTILINE).strip()
        sentences = [s.strip() for s in re.split(r"[.!?]\s+", clean_ans) if len(s.strip()) > 5]
        if not sentences:
            sentences = [clean_ans or "Primary clinical assertion"]

        claims: list[VerificationClaim] = []
        issues: list[str] = []

        if mode in ("always_accept", "realistic", "verified"):
            overall = OverallAssessmentStatus.VERIFIED
            support_score = 1.0
            issues = []
            for s in sentences[:3]:
                claims.append(
                    VerificationClaim(
                        claim=s,
                        status=ClaimVerificationStatus.SUPPORTED,
                        evidence=primary_pid,
                        reasoning=f"Directly corroborated by {primary_pid}.",
                    )
                )

        elif mode in ("always_reject", "not_verified"):
            overall = OverallAssessmentStatus.NOT_VERIFIED
            support_score = 0.0
            issues = ["None of the generated claims are supported by the provided evidence passages."]
            for s in sentences[:3]:
                claims.append(
                    VerificationClaim(
                        claim=s,
                        status=ClaimVerificationStatus.UNSUPPORTED,
                        evidence="NONE",
                        reasoning="Passage does not discuss this claim.",
                    )
                )

        elif mode == "contradicted":
            overall = OverallAssessmentStatus.NOT_VERIFIED
            support_score = 0.0
            issues = ["Claim explicitly contradicts clinical recommendations in the evidence."]
            for s in sentences[:2]:
                claims.append(
                    VerificationClaim(
                        claim=s,
                        status=ClaimVerificationStatus.CONTRADICTED,
                        evidence=primary_pid,
                        reasoning=f"Evidence in {primary_pid} states the opposite outcome.",
                    )
                )

        else:  # partially_verified
            overall = OverallAssessmentStatus.PARTIALLY_VERIFIED
            support_score = 0.5
            issues = ["Claim 2 lacks explicit evidence citation."]
            if len(sentences) >= 1:
                claims.append(
                    VerificationClaim(
                        claim=sentences[0],
                        status=ClaimVerificationStatus.SUPPORTED,
                        evidence=primary_pid,
                        reasoning="Directly substantiated by passage.",
                    )
                )
            if len(sentences) >= 2:
                claims.append(
                    VerificationClaim(
                        claim=sentences[1],
                        status=ClaimVerificationStatus.UNSUPPORTED,
                        evidence="NONE",
                        reasoning="No supporting evidence found in context.",
                    )
                )
            else:
                claims.append(
                    VerificationClaim(
                        claim="Secondary implied finding",
                        status=ClaimVerificationStatus.PARTIALLY_SUPPORTED,
                        evidence=primary_pid,
                        reasoning="Partially inferred but lacks quantitative data.",
                    )
                )

        # Build output string
        claims_str = "\n".join(
            f"- Claim: {c.claim} | Status: {c.status.value} | Evidence: {c.evidence} | Reasoning: {c.reasoning}"
            for c in claims
        )
        issues_str = "; ".join(issues) if issues else "None"
        formatted_output = (
            f"CLAIMS:\n{claims_str}\n\n"
            f"OVERALL_ASSESSMENT: {overall.value}\n"
            f"SUPPORT_SCORE: {support_score:.2f}\n"
            f"ISSUES: {issues_str}"
        )

        input_tokens = len(question.split()) + len(answer.split()) + sum(len(p.split()) for p in passages)
        output_tokens = len(formatted_output.split())
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0 + self.simulated_latency_ms

        parsed = ParsedVerificationResult(
            claims=claims,
            overall_assessment=overall,
            support_score=support_score,
            issues=issues,
            raw_text=formatted_output,
        )

        return GenerationOutput(
            text=formatted_output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            generation_time_ms=elapsed_ms,
            metadata={
                "provider": "mock_verifier",
                "mode": mode,
                "parsed_verification": parsed.model_dump(),
            },
        )
