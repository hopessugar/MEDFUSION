"""Unit tests for the evaluation metrics, statistical tests, and performance tracker."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.answer_metrics import (
    compute_bleu,
    compute_exact_match,
    compute_f1,
    compute_rouge_l,
    normalize_text,
)
from src.evaluation.calibration import (
    compute_brier_score,
    compute_coverage_risk_curve,
    compute_ece,
)
from src.evaluation.grounding_metrics import (
    compute_citation_precision,
    compute_claim_support_rate,
    compute_unsupported_claim_rate,
)
from src.evaluation.performance import PerformanceTracker
from src.evaluation.statistical import (
    bonferroni_correction,
    bootstrap_confidence_interval,
    mcnemar_test,
    paired_bootstrap_test,
)
from src.models.schemas import (
    AtomicClaim,
    Citation,
    ClaimSupportStatus,
    DecisionType,
    MedicalQuestion,
    PipelineDecision,
    PipelineResult,
    RetrievedPassage,
    VerificationResult,
)


def test_answer_metrics_exact_match() -> None:
    """Test text normalization and exact match computation."""
    assert normalize_text("The Patient Has Hypertension!") == "patient has hypertension"

    preds = ["Metformin 500mg daily", "Aspirin 81mg"]
    refs = ["Metformin 500mg daily", "Aspirin 100mg"]

    em = compute_exact_match(preds, refs)
    assert em == 0.5


def test_answer_metrics_f1_rouge_bleu() -> None:
    """Test F1, ROUGE-L, and BLEU calculations."""
    preds = ["Metformin is used as first-line treatment for type 2 diabetes."]
    refs = ["Metformin is the first-line medication for type 2 diabetes."]

    f1 = compute_f1(preds, refs)
    assert 0.7 <= f1 <= 1.0

    rouge_l = compute_rouge_l(preds, refs)
    assert 0.7 <= rouge_l <= 1.0

    bleu = compute_bleu(preds, refs)
    assert 0.0 <= bleu <= 1.0


def test_calibration_ece_and_brier() -> None:
    """Test ECE and Brier score metrics."""
    confs = [0.9, 0.9, 0.1, 0.1]
    labels = [1, 1, 0, 0]

    ece = compute_ece(confs, labels, n_bins=10)
    assert 0.0 <= ece <= 0.2

    brier = compute_brier_score(confs, labels)
    assert brier < 0.05

    # Coverage-risk curve
    curve = compute_coverage_risk_curve(confs, labels, thresholds=[0.0, 0.5, 0.8])
    assert len(curve["coverage"]) == 3
    assert len(curve["risk"]) == 3


def test_grounding_metrics() -> None:
    """Test claim support rate, unsupported rate, and citation precision."""
    c1 = AtomicClaim(
        claim_id="c1",
        claim_text="Lisinopril lowers systemic blood pressure.",
        citations=[Citation(claim_text="Lisinopril lowers blood pressure", passage_id="p1")],
        support_status=ClaimSupportStatus.SUPPORTED,
    )
    c2 = AtomicClaim(
        claim_id="c2",
        claim_text="Lisinopril prevents diabetic renal decline.",
        citations=[Citation(claim_text="Lisinopril prevents renal decline", passage_id="p1")],
        support_status=ClaimSupportStatus.SUPPORTED,
    )
    c3 = AtomicClaim(
        claim_id="c3",
        claim_text="Lisinopril cures diabetes permanently.",
        citations=[Citation(claim_text="Lisinopril cures diabetes", passage_id="p99")],
        support_status=ClaimSupportStatus.UNSUPPORTED,
    )

    res = PipelineResult(
        question=MedicalQuestion(question_id="q1", question="What treats hypertension?"),
        retrieved_passages=[
            RetrievedPassage(passage_id="p1", doc_id="d1", text="Lisinopril guideline", score=0.9)
        ],
        decision=PipelineDecision(
            decision=DecisionType.ACCEPT,
            score=0.9,
            reasoning="Claims verified by guidelines.",
        ),
        final_answer="Lisinopril is used.",
        verification_result=VerificationResult(
            verified=True,
            supported_claims=2,
            unsupported_claims=1,
            contradicted_claims=0,
            claim_results=[c1, c2, c3],
        ),
    )

    support_rate = compute_claim_support_rate([res])
    assert abs(support_rate - (2 / 3)) < 1e-4

    unsupported_rate = compute_unsupported_claim_rate([res])
    assert abs(unsupported_rate - (1 / 3)) < 1e-4

    cit_prec = compute_citation_precision([res])
    assert abs(cit_prec - (2 / 3)) < 1e-4


def test_statistical_tests() -> None:
    """Test bootstrap confidence interval, paired bootstrap, McNemar, and Bonferroni."""
    data = [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]

    point, lower, upper = bootstrap_confidence_interval(
        data, statistic_fn=lambda x: float(np.mean(x)), n_bootstrap=200
    )
    assert 0.6 <= point <= 0.9
    assert lower <= point <= upper

    # Paired bootstrap test
    p1 = [1, 1, 1, 1, 1, 1, 0, 1, 1, 1]
    p2 = [0, 0, 1, 0, 1, 0, 0, 0, 1, 0]
    boot_res = paired_bootstrap_test(p1, p2, n_bootstrap=200)
    assert boot_res["mean_diff"] > 0.0
    assert 0.0 <= boot_res["p_value"] <= 1.0

    # McNemar test
    mcnemar_res = mcnemar_test(p1, p2)
    assert mcnemar_res["statistic"] >= 0.0
    assert 0.0 <= mcnemar_res["p_value"] <= 1.0

    # Bonferroni correction
    p_values = [0.01, 0.04, 0.20]
    sig = bonferroni_correction(p_values, alpha=0.05)
    assert len(sig) == 3
    assert sig[0] is True   # 0.01 <= 0.05 / 3 = 0.0167
    assert sig[1] is False  # 0.04 > 0.0167
    assert sig[2] is False  # 0.20 > 0.0167


def test_performance_tracker() -> None:
    """Test PerformanceTracker latency recordings and summary percentiles."""
    tracker = PerformanceTracker()

    tracker.start_timer("retrieval")
    tracker.stop_timer("retrieval")

    tracker.start_timer("generation")
    tracker.stop_timer("generation")

    stats = tracker.get_latency_stats()
    assert "retrieval" in stats
    assert "generation" in stats
    assert stats["retrieval"]["count"] == 1
    assert stats["generation"]["count"] == 1
