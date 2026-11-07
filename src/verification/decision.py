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
        """Make an accept/revise/abstain decision.

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

        # Compute composite score
        composite_score = self._compute_composite_score(
            answer, verification, num_evidence_passages
        )
        
        # Calibrate the score
        calibrated_score = self.calibration.calibrate(composite_score)
        
        reasoning_parts.append(
            f"Raw score: {composite_score:.3f}, "
            f"Calibrated score: {calibrated_score:.3f}"
        )
        reasoning_parts.append(
            f"Verification: {verification.supported_claims} supported, "
            f"{verification.unsupported_claims} unsupported, "
            f"{verification.contradicted_claims} contradicted"
        )

        # Check for contradictions (always abstain if any contradicted claims)
        if verification.contradicted_claims > 0:
            reasoning_parts.append("Contradicted claims detected")
            return PipelineDecision(
                decision=DecisionType.ABSTAIN,
                score=calibrated_score,
                reasoning=" | ".join(reasoning_parts),
                revision_count=current_revision_count,
            )

        # Decision based on calibrated score
        if calibrated_score >= self.config.accept_min_score:
            decision = DecisionType.ACCEPT
            reasoning_parts.append(
                f"Score {calibrated_score:.3f} >= accept threshold "
                f"{self.config.accept_min_score}"
            )
        elif calibrated_score >= self.config.revise_min_score:
            # Can we still revise?
            if current_revision_count < self.config.abstention.max_revision_attempts:
                decision = DecisionType.REVISE
                reasoning_parts.append(
                    f"Score {calibrated_score:.3f} in revision range, "
                    f"revision {current_revision_count + 1}/"
                    f"{self.config.abstention.max_revision_attempts}"
                )
            else:
                # Exhausted revision attempts
                if calibrated_score >= self.config.abstain_below:
                    decision = DecisionType.ACCEPT
                    reasoning_parts.append(
                        "Revision attempts exhausted, accepting with lower confidence"
                    )
                else:
                    decision = DecisionType.ABSTAIN
                    reasoning_parts.append(
                        "Revision attempts exhausted, score too low to accept"
                    )
        else:
            decision = DecisionType.ABSTAIN
            reasoning_parts.append(
                f"Score {calibrated_score:.3f} < abstain threshold "
                f"{self.config.abstain_below}"
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

        Combines:
        - Verification support ratio (weight: 0.5)
        - Generator confidence (weight: 0.2, treated with skepticism)
        - Evidence availability signal (weight: 0.15)
        - Citation quality (weight: 0.15)
        """
        # Verification support ratio
        total_claims = len(verification.claim_results)
        if total_claims > 0:
            support_ratio = verification.overall_score
        else:
            support_ratio = 0.0

        # Generator self-reported confidence is known to exhibit severe overconfidence
        # in medical LLMs (Kadavath et al., 2022). We cap at 0.90 and discount by 0.80
        # (maximum contribution = 0.72) to prioritize evidence-grounded verification.
        gen_confidence = min(answer.confidence or 0.5, 0.9) * 0.8

        # Evidence availability
        evidence_signal = min(num_evidence / 5.0, 1.0)

        # Citation quality from verification
        if total_claims > 0:
            citation_quality = (
                verification.supported_claims / total_claims
            )
        else:
            citation_quality = 0.0

        composite = (
            0.50 * support_ratio
            + 0.20 * gen_confidence
            + 0.15 * evidence_signal
            + 0.15 * citation_quality
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

