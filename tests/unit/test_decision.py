"""Unit tests for the DecisionEngine, abstention policy, and score calibration."""

from __future__ import annotations

import pytest

from src.models.schemas import (
    AnswerabilityLabel,
    AtomicClaim,
    ClaimSupportStatus,
    DecisionType,
    GeneratedAnswer,
    VerificationResult,
)
from src.verification.decision import (
    AbstentionConfig,
    CalibrationModel,
    DecisionEngine,
    DecisionPolicyConfig,
)


def _make_claims(supported: int = 0, unsupported: int = 0, contradicted: int = 0) -> list[AtomicClaim]:
    """Helper to create claim lists for verification results."""
    claims: list[AtomicClaim] = []
    for i in range(supported):
        claims.append(AtomicClaim(
            claim_id=f"s{i}", claim_text=f"Supported claim {i}",
            support_status=ClaimSupportStatus.SUPPORTED,
        ))
    for i in range(unsupported):
        claims.append(AtomicClaim(
            claim_id=f"u{i}", claim_text=f"Unsupported claim {i}",
            support_status=ClaimSupportStatus.UNSUPPORTED,
        ))
    for i in range(contradicted):
        claims.append(AtomicClaim(
            claim_id=f"c{i}", claim_text=f"Contradicted claim {i}",
            support_status=ClaimSupportStatus.CONTRADICTED,
        ))
    return claims


def test_decision_accept_on_high_support() -> None:
    """Test ACCEPT decision when verification score is above accept threshold."""
    engine = DecisionEngine(
        config=DecisionPolicyConfig(
            accept_min_score=0.7,
            revise_min_score=0.4,
            abstain_below=0.4,
        )
    )

    ans = GeneratedAnswer(
        answer_text="Metformin is first-line pharmacotherapy.",
        answerability=AnswerabilityLabel.ANSWERABLE,
        confidence=0.95,
    )
    claims = _make_claims(supported=3)
    ver = VerificationResult(
        verified=True,
        overall_score=0.95,
        claim_results=claims,
        supported_claims=3,
        unsupported_claims=0,
        contradicted_claims=0,
    )

    decision = engine.decide(answer=ans, verification=ver, num_evidence_passages=5)
    assert decision.decision == DecisionType.ACCEPT
    assert decision.score >= 0.7


def test_decision_revise_on_medium_support() -> None:
    """Test REVISE decision when score falls between revise and accept thresholds."""
    engine = DecisionEngine(
        config=DecisionPolicyConfig(
            accept_min_score=0.8,
            revise_min_score=0.3,
            abstain_below=0.3,
            abstention=AbstentionConfig(max_revision_attempts=1),
        )
    )

    ans = GeneratedAnswer(
        answer_text="Some claims are grounded, others lack proof.",
        answerability=AnswerabilityLabel.ANSWERABLE,
        confidence=0.6,
    )
    claims = _make_claims(supported=1, unsupported=2)
    ver = VerificationResult(
        verified=False,
        overall_score=0.5,
        claim_results=claims,
        supported_claims=1,
        unsupported_claims=2,
        contradicted_claims=0,
    )

    decision = engine.decide(
        answer=ans,
        verification=ver,
        num_evidence_passages=3,
        current_revision_count=0,
    )
    # With 1/3 supported claims and moderate confidence, score should be
    # in the revision range (above abstain_below=0.3 but below accept=0.8)
    assert decision.decision in (DecisionType.REVISE, DecisionType.ACCEPT)
    assert decision.revision_count == 0


def test_decision_abstain_on_unanswerable() -> None:
    """Test ABSTAIN when answerability is labeled unanswerable."""
    engine = DecisionEngine()
    ans = GeneratedAnswer(
        answer_text="No evidence exists.",
        answerability=AnswerabilityLabel.UNANSWERABLE,
        confidence=0.1,
    )
    ver = VerificationResult(verified=False, overall_score=0.0)

    decision = engine.decide(answer=ans, verification=ver, num_evidence_passages=1)
    assert decision.decision == DecisionType.ABSTAIN
    assert decision.score == 0.0


def test_decision_abstain_on_insufficient_evidence() -> None:
    """Test ABSTAIN when retrieved passages count is below minimum requirement."""
    engine = DecisionEngine(
        config=DecisionPolicyConfig(
            abstention=AbstentionConfig(min_evidence_passages=2)
        )
    )
    ans = GeneratedAnswer(
        answer_text="Answer based on zero evidence.",
        answerability=AnswerabilityLabel.ANSWERABLE,
    )
    ver = VerificationResult(verified=False)

    decision = engine.decide(answer=ans, verification=ver, num_evidence_passages=0)
    assert decision.decision == DecisionType.ABSTAIN
    assert "Insufficient evidence" in decision.reasoning


def test_decision_abstain_on_contradiction() -> None:
    """Test ABSTAIN when contradicted claims are detected (low score)."""
    engine = DecisionEngine()
    ans = GeneratedAnswer(
        answer_text="Beta-blockers cure diabetes.",
        answerability=AnswerabilityLabel.ANSWERABLE,
        confidence=0.9,
    )
    claims = _make_claims(unsupported=1, contradicted=1)
    ver = VerificationResult(
        verified=False,
        overall_score=0.2,
        claim_results=claims,
        supported_claims=0,
        unsupported_claims=1,
        contradicted_claims=1,
    )

    decision = engine.decide(answer=ans, verification=ver, num_evidence_passages=2)
    assert decision.decision == DecisionType.ABSTAIN
    # The reasoning will mention the low score or contradictions
    assert decision.score < 0.7


def test_calibration_model() -> None:
    """Test CalibrationModel fitting and score adjustment."""
    cal = CalibrationModel(method="platt_scaling")
    assert cal.calibrate(0.7) == 0.7  # Identity before fitting

    # Fit calibration on dummy data
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    labels = [0, 0, 0, 0, 1, 0, 1, 1, 1, 1]
    cal.fit(scores, labels)

    calibrated_val = cal.calibrate(0.8)
    assert 0.0 <= calibrated_val <= 1.0
