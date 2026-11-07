"""Calibration and selective classification metrics for MedFusion-RAFT.

Implements Expected Calibration Error (ECE), Brier score, risk-coverage trade-offs,
and reliability diagram visualization for clinical risk management.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


def compute_ece(
    confidences: Sequence[float], correctness: Sequence[int], n_bins: int = 15
) -> float:
    """Compute Expected Calibration Error (ECE) across confidence bins.

    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

    Args:
        confidences: Predicted confidence or probability scores in [0.0, 1.0].
        correctness: Binary correctness ground truth (1 = correct, 0 = incorrect).
        n_bins: Number of equal-width calibration bins.

    Returns:
        Expected calibration error in [0.0, 1.0].
    """
    if len(confidences) == 0 or len(correctness) == 0:
        return 0.0

    confs = np.clip(np.asarray(confidences, dtype=np.float64), 0.0, 1.0)
    labels = np.asarray(correctness, dtype=np.int32)
    n = len(confs)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (confs >= bin_lower) & (confs <= bin_upper)
        else:
            in_bin = (confs >= bin_lower) & (confs < bin_upper)

        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(labels[in_bin])
            bin_conf = np.mean(confs[in_bin])
            ece += (bin_size / n) * np.abs(bin_acc - bin_conf)

    return float(ece)


def compute_brier_score(
    probabilities: Sequence[float], outcomes: Sequence[int]
) -> float:
    """Compute Brier Score measuring mean squared probability calibration error.

    Brier = (1/N) * sum_i (p_i - y_i)^2

    Args:
        probabilities: Predicted probabilities in [0.0, 1.0].
        outcomes: Binary outcomes (0 or 1).

    Returns:
        Brier score in [0.0, 1.0]. Lower values indicate superior calibration.
    """
    if len(probabilities) == 0 or len(outcomes) == 0:
        return 0.0

    probs = np.clip(np.asarray(probabilities, dtype=np.float64), 0.0, 1.0)
    y = np.asarray(outcomes, dtype=np.float64)
    return float(np.mean((probs - y) ** 2))


def compute_coverage_risk_curve(
    scores: Sequence[float],
    correctness: Sequence[int],
    thresholds: Optional[Sequence[float]] = None,
) -> dict[str, list[float]]:
    """Compute selective classification risk-coverage curve across rejection thresholds.

    Coverage is the proportion of predictions retained (score >= tau).
    Selective Risk is the empirical error rate among retained predictions.

    Args:
        scores: Model decision scores or calibrated confidences.
        correctness: Binary correctness labels (1 = correct, 0 = error).
        thresholds: Candidate decision thresholds. Defaults to 50 linearly spaced values.

    Returns:
        Dictionary mapping 'thresholds', 'coverage', and 'risk' lists.
    """
    if len(scores) == 0 or len(correctness) == 0:
        return {"thresholds": [], "coverage": [], "risk": []}

    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(correctness, dtype=np.int32)
    n = len(s)

    if thresholds is None:
        threshold_vals = np.linspace(0.0, 1.0, 50)
    else:
        threshold_vals = np.asarray(thresholds, dtype=np.float64)

    cov_list: list[float] = []
    risk_list: list[float] = []
    thresh_list: list[float] = []

    for t in threshold_vals:
        accepted = s >= t
        num_accepted = int(np.sum(accepted))
        cov = num_accepted / n
        if num_accepted > 0:
            error_count = int(np.sum(y[accepted] == 0))
            risk = error_count / num_accepted
        else:
            risk = 0.0

        thresh_list.append(float(t))
        cov_list.append(float(cov))
        risk_list.append(float(risk))

    return {
        "thresholds": thresh_list,
        "coverage": cov_list,
        "risk": risk_list,
    }


def compute_selective_risk(
    scores: Sequence[float], correctness: Sequence[int], coverage: float
) -> float:
    """Compute empirical risk at a designated target coverage level.

    Args:
        scores: Confidence scores.
        correctness: Correctness labels (1 = correct, 0 = error).
        coverage: Target coverage fraction in (0.0, 1.0].

    Returns:
        Empirical error rate over the top-coverage fraction of predictions.
    """
    if len(scores) == 0 or len(correctness) == 0:
        return 0.0

    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(correctness, dtype=np.int32)
    n = len(s)

    target_k = max(1, min(n, int(np.ceil(coverage * n))))
    sorted_indices = np.argsort(s)[::-1]  # descending order
    top_indices = sorted_indices[:target_k]

    errors = np.sum(y[top_indices] == 0)
    return float(errors / target_k)


def plot_reliability_diagram(
    confidences: Sequence[float],
    correctness: Sequence[int],
    output_path: str,
    n_bins: int = 10,
) -> None:
    """Plot reliability diagram with calibration bars and save to file.

    Args:
        confidences: Confidence values in [0, 1].
        correctness: Correctness values (0 or 1).
        output_path: Output image destination path (.png or .pdf).
        n_bins: Number of histogram bins.
    """
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    if len(confidences) == 0 or len(correctness) == 0:
        logger.warning("Empty data provided to plot_reliability_diagram")
        return

    confs = np.clip(np.asarray(confidences, dtype=np.float64), 0.0, 1.0)
    labels = np.asarray(correctness, dtype=np.int32)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2.0
    bin_accs: list[float] = []
    bin_confs: list[float] = []
    bin_counts: list[int] = []

    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confs >= low) & (confs <= high) if i == n_bins - 1 else (confs >= low) & (confs < high)
        count = int(np.sum(in_bin))
        bin_counts.append(count)
        if count > 0:
            bin_accs.append(float(np.mean(labels[in_bin])))
            bin_confs.append(float(np.mean(confs[in_bin])))
        else:
            bin_accs.append(0.0)
            bin_confs.append(float(bin_centers[i]))

    ece = compute_ece(confidences, correctness, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    ax.bar(
        bin_centers,
        bin_accs,
        width=1.0 / n_bins,
        alpha=0.6,
        color="#2b5c8f",
        edgecolor="black",
        label=f"Observed Accuracy (ECE={ece:.3f})",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Empirical Accuracy")
    ax.set_title("Reliability Diagram")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.5)

    dest = Path(output_path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(dest, dpi=300)
    plt.close(fig)
    logger.info("Saved reliability diagram to %s", dest)
