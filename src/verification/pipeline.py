"""Verification pipeline combining citation validation, entailment, and LLM verification."""
from __future__ import annotations

import logging
import time
from typing import Optional

from pydantic import BaseModel, Field

from src.models.schemas import (
    AtomicClaim,
    ClaimSupportStatus,
    GeneratedAnswer,
    RetrievedPassage,
    VerificationResult,
)
from src.models.base import BaseVerifierProvider
from src.verification.claim_extractor import ClaimExtractor, ClaimExtractionConfig
from src.verification.citation_validator import (
    CitationValidator,
    CitationValidationConfig,
    OverallCitationResult,
)
from src.verification.entailment import (
    EntailmentChecker,
    EntailmentConfig,
    MockEntailmentChecker,
)

logger = logging.getLogger(__name__)


class VerificationPipelineConfig(BaseModel):
    """Configuration for the full verification pipeline."""
    use_citation_validation: bool = True
    use_entailment_checking: bool = True
    use_llm_verification: bool = True
    require_all_claims_supported: bool = False
    min_support_ratio: float = 0.6
    max_unsupported_claims: int = 2
    max_contradicted_claims: int = 0
    claim_config: ClaimExtractionConfig = Field(
        default_factory=ClaimExtractionConfig
    )
    citation_config: CitationValidationConfig = Field(
        default_factory=CitationValidationConfig
    )
    entailment_config: EntailmentConfig = Field(
        default_factory=EntailmentConfig
    )


class VerificationPipeline:
    """Full verification pipeline that combines multiple verification strategies.

    Pipeline stages:
    1. Extract atomic claims from the generated answer
    2. Validate citations (deterministic check)
    3. Check entailment using NLI model (deterministic check)
    4. LLM-based verification using Qwen verifier (optional)
    5. Aggregate results into a final VerificationResult
    """

    def __init__(
        self,
        config: Optional[VerificationPipelineConfig] = None,
        verifier_provider: Optional[BaseVerifierProvider] = None,
        entailment_checker: Optional[EntailmentChecker] = None,
    ) -> None:
        self.config = config or VerificationPipelineConfig()
        self.claim_extractor = ClaimExtractor(self.config.claim_config)
        self.citation_validator = CitationValidator(self.config.citation_config)
        self.entailment_checker = entailment_checker
        self.verifier_provider = verifier_provider

        # Initialize entailment checker if not provided
        if (
            self.config.use_entailment_checking
            and self.entailment_checker is None
        ):
            logger.warning(
                "No entailment checker provided; falling back to MockEntailmentChecker. "
                "For production use, pass an EntailmentChecker instance with a real NLI model."
            )
            self.entailment_checker = MockEntailmentChecker(
                self.config.entailment_config
            )

    def verify(
        self,
        question: str,
        answer: GeneratedAnswer,
        passages: list[RetrievedPassage],
    ) -> VerificationResult:
        """Run the full verification pipeline.

        Args:
            question: The original medical question.
            answer: The generated answer to verify.
            passages: The retrieved evidence passages.

        Returns:
            VerificationResult with claim-level analysis.
        """
        start_time = time.perf_counter()

        # Stage 1: Extract atomic claims
        logger.info("Stage 1: Extracting atomic claims")
        claims = self.claim_extractor.extract_claims(answer.answer_text)
        
        if not claims:
            logger.warning("No claims extracted from answer")
            elapsed = (time.perf_counter() - start_time) * 1000
            return VerificationResult(
                verified=False,
                overall_score=0.0,
                claim_results=[],
                supported_claims=0,
                unsupported_claims=0,
                contradicted_claims=0,
                verification_reasoning="No verifiable claims found in the answer.",
                verification_time_ms=elapsed,
                verifier_model_id="pipeline",
            )

        # Stage 2: Citation validation
        citation_result: Optional[OverallCitationResult] = None
        if self.config.use_citation_validation:
            logger.info("Stage 2: Validating citations")
            citation_result = self.citation_validator.validate(claims, passages)
            
            # Update claims with citation validation info
            for claim, cit_res in zip(claims, citation_result.claim_results):
                if not cit_res.is_valid and not cit_res.cited_passage_ids:
                    claim.support_status = ClaimSupportStatus.UNSUPPORTED

        # Stage 3: Entailment checking
        if self.config.use_entailment_checking and self.entailment_checker is not None:
            logger.info("Stage 3: Checking entailment")
            for claim in claims:
                # Get the passages cited by this claim, or all passages if none cited
                cited_ids = {c.passage_id for c in claim.citations}
                relevant_passages = (
                    [p for p in passages if p.passage_id in cited_ids]
                    if cited_ids
                    else passages[:5]  # Use top-5 if no specific citations
                )
                
                if relevant_passages:
                    status, score, _ = self.entailment_checker.check_claim_against_passages(
                        claim, relevant_passages
                    )
                    claim.support_status = status
                    claim.entailment_score = score
                else:
                    claim.support_status = ClaimSupportStatus.INSUFFICIENT_EVIDENCE
                    claim.entailment_score = 0.0

        # Stage 4: LLM-based verification (optional)
        llm_verification_reasoning: Optional[str] = None
        if (
            self.config.use_llm_verification
            and self.verifier_provider is not None
            and self.verifier_provider.is_loaded
        ):
            logger.info("Stage 4: LLM-based verification")
            try:
                evidence_texts = [p.text for p in passages]
                llm_output = self.verifier_provider.verify_answer(
                    question=question,
                    answer=answer.answer_text,
                    evidence=evidence_texts,
                )
                llm_verification_reasoning = llm_output.text
                
                # Parse LLM verification output and cross-reference with
                # deterministic checks (don't blindly trust the LLM)
                self._integrate_llm_verification(claims, llm_output.text)
            except Exception as e:
                logger.warning("LLM verification failed: %s", e)
                llm_verification_reasoning = f"LLM verification error: {e}"

        # Stage 5: Aggregate results
        logger.info("Stage 5: Aggregating verification results")
        supported = sum(
            1 for c in claims
            if c.support_status == ClaimSupportStatus.SUPPORTED
        )
        partially_supported = sum(
            1 for c in claims
            if c.support_status == ClaimSupportStatus.PARTIALLY_SUPPORTED
        )
        unsupported = sum(
            1 for c in claims
            if c.support_status == ClaimSupportStatus.UNSUPPORTED
        )
        contradicted = sum(
            1 for c in claims
            if c.support_status == ClaimSupportStatus.CONTRADICTED
        )
        insufficient = sum(
            1 for c in claims
            if c.support_status == ClaimSupportStatus.INSUFFICIENT_EVIDENCE
        )

        total = len(claims)
        support_ratio = (supported + 0.5 * partially_supported) / total if total > 0 else 0.0
        
        # Determine overall verification status
        verified = self._determine_verified(
            supported=supported,
            partially_supported=partially_supported,
            unsupported=unsupported,
            contradicted=contradicted,
            total=total,
        )

        # Build reasoning
        reasoning_parts: list[str] = []
        reasoning_parts.append(
            f"Claims: {total} total, {supported} supported, "
            f"{partially_supported} partially supported, "
            f"{unsupported} unsupported, {contradicted} contradicted, "
            f"{insufficient} insufficient evidence."
        )
        if citation_result:
            reasoning_parts.append(
                f"Citations: {citation_result.valid_citations}/{citation_result.total_citations} valid "
                f"(precision={citation_result.citation_precision:.3f})."
            )
        if llm_verification_reasoning:
            reasoning_parts.append(
                f"LLM verification: {llm_verification_reasoning[:200]}"
            )

        elapsed = (time.perf_counter() - start_time) * 1000

        return VerificationResult(
            verified=verified,
            overall_score=support_ratio,
            claim_results=claims,
            supported_claims=supported,
            unsupported_claims=unsupported,
            contradicted_claims=contradicted,
            verification_reasoning=" | ".join(reasoning_parts),
            verification_time_ms=elapsed,
            verifier_model_id=(
                self.verifier_provider.config.model_id
                if self.verifier_provider
                else "deterministic_pipeline"
            ),
        )

    def _determine_verified(
        self,
        supported: int,
        partially_supported: int,
        unsupported: int,
        contradicted: int,
        total: int,
    ) -> bool:
        """Determine whether the answer passes verification."""
        if contradicted > self.config.max_contradicted_claims:
            return False
        
        if unsupported > self.config.max_unsupported_claims:
            return False
        
        if self.config.require_all_claims_supported:
            return unsupported == 0 and contradicted == 0
        
        support_ratio = (
            (supported + 0.5 * partially_supported) / total
            if total > 0
            else 0.0
        )
        return support_ratio >= self.config.min_support_ratio

    def _integrate_llm_verification(
        self,
        claims: list[AtomicClaim],
        llm_output: str,
    ) -> None:
        """Integrate LLM verification output with existing claim statuses.

        The LLM output is treated as supplementary evidence, not as
        overriding the deterministic checks. If the LLM and deterministic
        checks disagree, we prefer the deterministic result but log the
        disagreement.
        """
        # Parse the LLM output for claim-level assessments
        import re
        
        claim_pattern = re.compile(
            r'Claim:\s*(.+?)\s*\|\s*Status:\s*(\w+)',
            re.IGNORECASE,
        )
        
        llm_assessments: dict[str, str] = {}
        for match in claim_pattern.finditer(llm_output):
            claim_text_snippet = match.group(1).strip()[:50].lower()
            status_str = match.group(2).strip().lower()
            llm_assessments[claim_text_snippet] = status_str
        
        # Cross-reference with existing claims
        for claim in claims:
            claim_snippet = claim.claim_text[:50].lower()
            if claim_snippet in llm_assessments:
                llm_status = llm_assessments[claim_snippet]
                
                # Map LLM status string to enum
                status_map = {
                    'supported': ClaimSupportStatus.SUPPORTED,
                    'partially_supported': ClaimSupportStatus.PARTIALLY_SUPPORTED,
                    'unsupported': ClaimSupportStatus.UNSUPPORTED,
                    'contradicted': ClaimSupportStatus.CONTRADICTED,
                    'not_verified': ClaimSupportStatus.UNSUPPORTED,
                }
                
                llm_claim_status = status_map.get(llm_status)
                
                if llm_claim_status and claim.support_status != llm_claim_status:
                    logger.info(
                        "Disagreement for claim '%s': "
                        "deterministic=%s, LLM=%s (keeping deterministic)",
                        claim.claim_text[:50],
                        claim.support_status,
                        llm_claim_status,
                    )

