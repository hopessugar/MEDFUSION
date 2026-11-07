"""Evaluation suite for medical QA accuracy, grounding, retrieval metrics, and selective risk."""

from src.evaluation.answer_metrics import (
    compute_accuracy,
    compute_bleu,
    compute_exact_match,
    compute_f1,
    compute_rouge_l,
    evaluate_answers,
    normalize_text,
)
from src.evaluation.calibration import (
    compute_brier_score,
    compute_coverage_risk_curve,
    compute_ece,
    compute_selective_risk,
    plot_reliability_diagram,
)
from src.evaluation.grounding_metrics import (
    compute_citation_precision,
    compute_citation_recall,
    compute_claim_support_rate,
    compute_contradiction_rate,
    compute_unsupported_claim_rate,
    evaluate_grounding,
)
from src.evaluation.performance import PerformanceTracker
from src.evaluation.runner import EvaluationRunner
from src.evaluation.statistical import (
    bonferroni_correction,
    bootstrap_confidence_interval,
    compute_confidence_intervals_for_metrics,
    mcnemar_test,
    paired_bootstrap_test,
)

__all__ = [
    # Answer metrics
    "compute_accuracy",
    "compute_bleu",
    "compute_exact_match",
    "compute_f1",
    "compute_rouge_l",
    "evaluate_answers",
    "normalize_text",
    # Grounding metrics
    "compute_citation_precision",
    "compute_citation_recall",
    "compute_claim_support_rate",
    "compute_contradiction_rate",
    "compute_unsupported_claim_rate",
    "evaluate_grounding",
    # Calibration & Selective Risk
    "compute_brier_score",
    "compute_coverage_risk_curve",
    "compute_ece",
    "compute_selective_risk",
    "plot_reliability_diagram",
    # Statistical significance
    "bonferroni_correction",
    "bootstrap_confidence_interval",
    "compute_confidence_intervals_for_metrics",
    "mcnemar_test",
    "paired_bootstrap_test",
    # Performance profiling
    "PerformanceTracker",
    # Evaluation Runner
    "EvaluationRunner",
]
