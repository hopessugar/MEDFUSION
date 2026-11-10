"""Abstention policy and decision engine for MedFusion-RAFT."""
from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from src.models.schemas import (
    DecisionType,
    GeneratedAnswer,
    PipelineDecision,
    VerificationResult,
    AnswerabilityLabel,
)

logger = logging.getLogger(__name__)


class AbstentionConfig(BaseModel):
    """Configuration for the abstention policy."""
    enabled: bool = True
    confidence_threshold: float = 0.5
    min_evidence_passages: int = 1
    max_revision_attempts: int = 1
    calibration_method: str = "platt_scaling"  # platt_scaling, isotonic, none


class DecisionPolicyConfig(BaseModel):
    """Configuration for the decision policy."""
    accept_min_score: float = 0.7
    revise_min_score: float = 0.4
    abstain_below: float = 0.4
    # Medical safety thresholds
    max_unsupported_ratio: float = 0.3  # Max 30% unsupported claims
    max_unsupported_absolute: int = 2   # Max 2 unsupported claims regardless of ratio
    allow_partially_supported: bool = True  # Allow partially supported in final answer
    require_critical_claims_supported: bool = True  # High-confidence claims must be supported
    abstention: AbstentionConfig = Field(default_factory=AbstentionConfig)


class CalibrationModel:
    """Calibrates raw confidence/support scores to probabilities.

    Uses Platt scaling (logistic regression) or isotonic regression
    fitted on validation data.
    """

    def __init__(self, method: str = "platt_scaling") -> None:
        self.method = method
        self._fitted = False
        self._platt_a: float = 1.0
        self._platt_b: float = 0.0
        self._isotonic_model = None

    def fit(self, scores: list[float], labels: list[int]) -> None:
        """Fit the calibration model on validation data.

        Args:
            scores: Raw model scores/confidences.
            labels: Binary labels (1 = correct, 0 = incorrect).
        """
        if len(scores) < 10:
            logger.warning(
                "Too few samples (%d) for calibration, using identity",
                len(scores),
            )
            return

        scores_arr = np.array(scores)
        labels_arr = np.array(labels)

        if self.method == "platt_scaling":
            self._fit_platt(scores_arr, labels_arr)
        elif self.method == "isotonic":
            self._fit_isotonic(scores_arr, labels_arr)
        else:
            logger.info("No calibration method; using raw scores")

        self._fitted = True

    def calibrate(self, score: float) -> float:
        """Calibrate a raw score to a probability.

        Args:
            score: Raw score in [0, 1].

        Returns:
            Calibrated probability in [0, 1].
        """
        if not self._fitted:
            return score

        if self.method == "platt_scaling":
            return self._platt_transform(score)
        elif self.method == "isotonic" and self._isotonic_model is not None:
            return float(
                self._isotonic_model.predict(np.array([[score]]))[0]
            )
        return score

    def _fit_platt(self, scores: np.ndarray, labels: np.ndarray) -> None:
        """Fit Platt scaling (logistic regression on scores)."""
        try:
            from sklearn.linear_model import LogisticRegression

            model = LogisticRegression(C=1.0, solver="lbfgs")
            model.fit(scores.reshape(-1, 1), labels)
            self._platt_a = float(model.coef_[0][0])
            self._platt_b = float(model.intercept_[0])
            logger.info(
                "Platt scaling fitted: a=%.4f, b=%.4f",
                self._platt_a,
                self._platt_b,
            )
        except ImportError:
            logger.warning("scikit-learn not available for Platt scaling")
        except Exception as e:
            logger.warning("Platt scaling fitting failed: %s", e)

    def _fit_isotonic(self, scores: np.ndarray, labels: np.ndarray) -> None:
        """Fit isotonic regression on scores."""
        try:
            from sklearn.isotonic import IsotonicRegression

            self._isotonic_model = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip"
            )
            self._isotonic_model.fit(scores, labels)
            logger.info("Isotonic regression fitted")
        except ImportError:
            logger.warning("scikit-learn not available for isotonic regression")
        except Exception as e:
            logger.warning("Isotonic regression fitting failed: %s", e)

    def _platt_transform(self, score: float) -> float:
        """Apply Platt scaling transform."""
        logit = self._platt_a * score + self._platt_b
        return 1.0 / (1.0 + np.exp(-logit))

    def save(self, path: str) -> None:
        """Save calibration parameters."""
        data = {
            "method": self.method,
            "fitted": self._fitted,
            "platt_a": self._platt_a,
            "platt_b": self._platt_b,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str) -> None:
        """Load calibration parameters."""
        with open(path) as f:
            data = json.load(f)
        self.method = data["method"]
        self._fitted = data["fitted"]
        self._platt_a = data.get("platt_a", 1.0)
        self._platt_b = data.get("platt_b", 0.0)


class DecisionEngine:
    """Makes accept/revise/abstain decisions based on verification results.

    Combines verification scores, answer confidence, evidence availability,
    and calibrated abstention thresholds to make the final decision.
    """

    def __init__(
        self,
        config: Optional[DecisionPolicyConfig] = None,
    ) -> None:
        self.config = config or DecisionPolicyConfig()
        self.calibration = CalibrationModel(
            method=self.config.abstention.calibration_method
        )

    def decide(
        self,
        answer: GeneratedAnswer,
        verification: VerificationResult,
        num_evidence_passages: int,
        current_revision_count: int = 0,
    ) -> PipelineDecision:
        """Make an accept/revise/abstain decision with medical safety-first policy.

        Decision logic prioritizes patient safety:
        1. Any contradicted claims → immediate ABSTAIN
        2. Excessive unsupported claims → ABSTAIN (configurable thresholds)
        3. Insufficient evidence → ABSTAIN
        4. Composite score with heavy weight on verification
        5. Revision opportunity if score in acceptable range

        Args:
            answer: The generated answer.
            verification: The verification result.
            num_evidence_passages: Number of retrieved evidence passages.
            current_revision_count: How many revisions have been attempted.

        Returns:
            PipelineDecision with the decision type, score, and reasoning.
        """
        reasoning_parts: list[str] = []

        # Check if the generator itself declared unanswerable
        if answer.answerability == AnswerabilityLabel.UNANSWERABLE:
            return PipelineDecision(
                decision=DecisionType.ABSTAIN,
                score=0.0,
                reasoning="Generator declared the question unanswerable based on available evidence.",
                revision_count=current_revision_count,
            )

        # Check minimum evidence requirement
        if (
            self.config.abstention.enabled
            and num_evidence_passages < self.config.abstention.min_evidence_passages
        ):
            return PipelineDecision(
                decision=DecisionType.ABSTAIN,
                score=0.1,
                reasoning=(
                    f"Insufficient evidence: {num_evidence_passages} passages "
                    f"retrieved (minimum: {self.config.abstention.min_evidence_passages})."
                ),
                revision_count=current_revision_count,
            )

        # MEDICAL SAFETY CHECK 1: Any contradicted claims → immediate ABSTAIN
        if verification.contradicted_claims > 0:
            reasoning_parts.append(
                f"SAFETY: {verification.contradicted_claims} contradicted claim(s) detected - "
                "cannot provide safe answer"
            )
            return PipelineDecision(
                decision=DecisionType.ABSTAIN,
                score=0.0,
                reasoning=" | ".join(reasoning_parts),
                revision_count=current_revision_count,
            )

        # MEDICAL SAFETY CHECK 2: Excessive unsupported claims
        total_claims = len(verification.claim_results)
        if total_claims > 0:
            unsupported_ratio = verification.unsupported_claims / total_claims
            
            # Check both ratio and absolute count
            if (
                unsupported_ratio > self.config.max_unsupported_ratio
                or verification.unsupported_claims > self.config.max_unsupported_absolute
            ):
                # If we haven't tried revision yet and are below max attempts, try revision
                if current_revision_count < self.config.abstention.max_revision_attempts:
                    reasoning_parts.append(
                        f"SAFETY: Too many unsupported claims ({verification.unsupported_claims}/{total_claims}, "
                        f"{unsupported_ratio:.1%}) - attempting revision"
                    )
                    return PipelineDecision(
                        decision=DecisionType.REVISE,
                        score=0.3,
                        reasoning=" | ".join(reasoning_parts),
                        revision_count=current_revision_count,
                    )
                else:
                    reasoning_parts.append(
                        f"SAFETY: Too many unsupported claims ({verification.unsupported_claims}/{total_claims}, "
                        f"{unsupported_ratio:.1%}) - revisions exhausted"
                    )
                    return PipelineDecision(
                        decision=DecisionType.ABSTAIN,
                        score=0.2,
                        reasoning=" | ".join(reasoning_parts),
                        revision_count=current_revision_count,
                    )

        # Compute composite score
        composite_score = self._compute_composite_score(
            answer, verification, num_evidence_passages
        )
        
        # Calibrate the score
        calibrated_score = self.calibration.calibrate(composite_score)
        
        reasoning_parts.append(
            f"Composite: {composite_score:.3f} → Calibrated: {calibrated_score:.3f}"
        )
        reasoning_parts.append(
            f"Claims: {verification.supported_claims}✓ {verification.unsupported_claims}✗ "
            f"{verification.contradicted_claims}⊗ (of {total_claims} total)"
        )

        # Detailed claim status breakdown
        from src.models.schemas import ClaimSupportStatus
        claim_statuses = {}
        for claim in verification.claim_results:
            status = claim.support_status.value
            claim_statuses[status] = claim_statuses.get(status, 0) + 1
        
        status_summary = ", ".join(
            f"{count} {status}" for status, count in sorted(claim_statuses.items())
        )
        reasoning_parts.append(f"Breakdown: {status_summary}")

        # Decision based on calibrated score with medical safety considerations
        if calibrated_score >= self.config.accept_min_score:
            decision = DecisionType.ACCEPT
            reasoning_parts.append(
                f"Score {calibrated_score:.3f} ≥ accept threshold {self.config.accept_min_score:.2f}"
            )
        elif calibrated_score >= self.config.revise_min_score:
            # Can we still revise?
            if current_revision_count < self.config.abstention.max_revision_attempts:
                decision = DecisionType.REVISE
                reasoning_parts.append(
                    f"Score {calibrated_score:.3f} in revision range "
                    f"[{self.config.revise_min_score:.2f}, {self.config.accept_min_score:.2f}), "
                    f"revision {current_revision_count + 1}/{self.config.abstention.max_revision_attempts}"
                )
            else:
                # Exhausted revision attempts
                if calibrated_score >= self.config.abstain_below:
                    decision = DecisionType.ACCEPT
                    reasoning_parts.append(
                        f"Revisions exhausted, accepting with moderate confidence "
                        f"(score {calibrated_score:.3f} ≥ floor {self.config.abstain_below:.2f})"
                    )
                else:
                    decision = DecisionType.ABSTAIN
                    reasoning_parts.append(
                        f"Revisions exhausted, score {calibrated_score:.3f} < floor {self.config.abstain_below:.2f}"
                    )
        else:
            decision = DecisionType.ABSTAIN
            reasoning_parts.append(
                f"Score {calibrated_score:.3f} < abstain threshold {self.config.abstain_below:.2f}"
            )

        return PipelineDecision(
            decision=decision,
            score=calibrated_score,
            reasoning=" | ".join(reasoning_parts),
            revision_count=current_revision_count,
        )

    def _compute_composite_score(
        self,
        answer: GeneratedAnswer,
        verification: VerificationResult,
        num_evidence: int,
    ) -> float:
        """Compute a composite confidence score from multiple signals.

        Medical safety-first scoring combines:
        - Verification support ratio with claim-level granularity (weight: 0.50)
        - Generator confidence, heavily discounted for overconfidence (weight: 0.20)
        - Evidence availability and quality signal (weight: 0.15)
        - Citation precision (weight: 0.15)

        Claim status value mapping:
        - SUPPORTED: 1.0 (full value)
        - PARTIALLY_SUPPORTED: 0.5 (half value)
        - INSUFFICIENT_EVIDENCE: 0.2 (small credit for non-contradicted)
        - UNSUPPORTED: 0.0 (no value)
        - CONTRADICTED: -0.5 (penalty)

        Args:
            answer: Generated answer with self-reported confidence.
            verification: Verification result with claim-level analysis.
            num_evidence: Number of retrieved evidence passages.

        Returns:
            Composite score in [0.0, 1.0].
        """
        from src.models.schemas import ClaimSupportStatus
        
        total_claims = len(verification.claim_results)
        
        # Component 1: Claim-level verification score with granular weighting
        if total_claims > 0:
            claim_value_sum = 0.0
            for claim in verification.claim_results:
                if claim.support_status == ClaimSupportStatus.SUPPORTED:
                    claim_value_sum += 1.0
                elif claim.support_status == ClaimSupportStatus.PARTIALLY_SUPPORTED:
                    claim_value_sum += 0.5
                elif claim.support_status == ClaimSupportStatus.INSUFFICIENT_EVIDENCE:
                    claim_value_sum += 0.2  # Small credit for not being contradicted
                elif claim.support_status == ClaimSupportStatus.UNSUPPORTED:
                    claim_value_sum += 0.0
                elif claim.support_status == ClaimSupportStatus.CONTRADICTED:
                    claim_value_sum -= 0.5  # Penalty for contradiction
            
            # Normalize to [0, 1] range
            # Max possible: total_claims * 1.0, Min possible: total_claims * -0.5
            # Shift and scale to [0, 1]
            normalized_claim_score = (claim_value_sum + 0.5 * total_claims) / (1.5 * total_claims)
            verification_score = float(np.clip(normalized_claim_score, 0.0, 1.0))
        else:
            verification_score = 0.0

        # Component 2: Generator self-reported confidence
        # Medical LLMs exhibit severe overconfidence (Kadavath et al., 2022; Tian et al., 2023).
        # We apply aggressive discounting: cap at 0.90, multiply by 0.75 (max contribution = 0.675)
        # to prioritize evidence-grounded verification over self-assessment.
        gen_confidence = min(answer.confidence or 0.5, 0.9) * 0.75

        # Component 3: Evidence availability and quality signal
        # Use logarithmic scaling to reduce impact of excessive passages
        # 1-2 passages: low signal, 3-5 passages: good, 6+ passages: diminishing returns
        evidence_signal = float(np.clip(np.log1p(num_evidence) / np.log1p(5.0), 0.0, 1.0))

        # Component 4: Citation precision (supported claims with valid citations)
        # This rewards explicit grounding beyond just verification
        if total_claims > 0:
            citation_quality = verification.supported_claims / total_claims
        else:
            citation_quality = 0.0

        # Weighted combination
        # Prioritize verification (50%), evidence-grounding (15% + 15% = 30%), then generator (20%)
        composite = (
            0.50 * verification_score      # Primary: claim verification
            + 0.20 * gen_confidence         # Secondary: discounted self-confidence
            + 0.15 * evidence_signal        # Tertiary: evidence availability
            + 0.15 * citation_quality       # Tertiary: citation precision
        )

        return float(np.clip(composite, 0.0, 1.0))

    def fit_calibration(
        self,
        scores: list[float],
        labels: list[int],
    ) -> None:
        """Fit calibration model on validation data."""
        self.calibration.fit(scores, labels)

    def save_calibration(self, path: str) -> None:
        """Save calibration model."""
        self.calibration.save(path)

    def load_calibration(self, path: str) -> None:
        """Load calibration model."""
        self.calibration.load(path)

