"""Statistical significance testing and bootstrap confidence estimation for MedFusion-RAFT.

Provides rigorous confidence interval estimation, paired bootstrap hypotheses testing,
McNemar's test for binary model superiority, and multiple testing corrections.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
from scipy import stats

from src.models.schemas import PipelineResult
from src.utils.logging import get_logger

logger = get_logger(__name__)


def bootstrap_confidence_interval(
    data: Sequence[Any],
    statistic_fn: Callable[[Sequence[Any]], float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Calculate non-parametric bootstrap confidence interval for an arbitrary metric.

    Args:
        data: Sequence of samples.
        statistic_fn: Function calculating target metric scalar from sample sub-sequence.
        n_bootstrap: Number of bootstrap iterations.
        ci: Confidence interval coverage in (0.0, 1.0).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper).
    """
    n = len(data)
    if n == 0:
        return 0.0, 0.0, 0.0

    point_estimate = float(statistic_fn(data))
    if n == 1 or n_bootstrap <= 1:
        return point_estimate, point_estimate, point_estimate

    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    boot_stats = np.empty(n_bootstrap, dtype=np.float64)

    # Convert data to indexable structure
    data_list = list(data)

    for i in range(n_bootstrap):
        sampled_indices = rng.choice(indices, size=n, replace=True)
        sample = [data_list[idx] for idx in sampled_indices]
        boot_stats[i] = statistic_fn(sample)

    alpha = 1.0 - ci
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    ci_lower = float(np.percentile(boot_stats, lower_pct))
    ci_upper = float(np.percentile(boot_stats, upper_pct))

    return point_estimate, ci_lower, ci_upper


def paired_bootstrap_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Conduct a paired two-sided bootstrap hypothesis test comparing two systems.

    Null hypothesis H0: Mean difference in performance = 0.

    Args:
        scores_a: Metric scores from system A.
        scores_b: Metric scores from system B.
        n_bootstrap: Number of bootstrap iterations.
        seed: Random seed.

    Returns:
        Dictionary with 'mean_diff', 'p_value', and 'effect_size' (Cohen's d).
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("scores_a and scores_b must have identical lengths")

    n = len(scores_a)
    if n == 0:
        return {"mean_diff": 0.0, "p_value": 1.0, "effect_size": 0.0}

    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    diffs = a - b
    observed_mean_diff = float(np.mean(diffs))

    std_diff = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    effect_size = (observed_mean_diff / std_diff) if std_diff > 1e-12 else 0.0

    # Under H0, center differences around 0
    centered_diffs = diffs - observed_mean_diff
    rng = np.random.default_rng(seed)

    boot_diffs = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        resampled = rng.choice(centered_diffs, size=n, replace=True)
        boot_diffs[i] = np.mean(resampled)

    # Two-sided empirical p-value
    abs_obs = abs(observed_mean_diff)
    extreme_count = np.sum(np.abs(boot_diffs) >= abs_obs)
    p_value = float((extreme_count + 1) / (n_bootstrap + 1))

    return {
        "mean_diff": round(observed_mean_diff, 4),
        "p_value": round(p_value, 5),
        "effect_size": round(effect_size, 4),
    }


def mcnemar_test(
    correct_a: Sequence[int], correct_b: Sequence[int]
) -> dict[str, float]:
    """Execute McNemar's test for paired binary classification outcomes.

    Assesses whether the disagreement frequencies between system A and system B
    deviate significantly from symmetry.

    Args:
        correct_a: Binary correctness flags for system A (1 = correct, 0 = error).
        correct_b: Binary correctness flags for system B.

    Returns:
        Dictionary containing 'statistic', 'p_value', 'b_count', and 'c_count'.
    """
    if len(correct_a) != len(correct_b):
        raise ValueError("Inputs must have identical length")

    ca = np.asarray(correct_a, dtype=np.int32)
    cb = np.asarray(correct_b, dtype=np.int32)

    # Discordant counts:
    # b: System A wrong (0), System B right (1)
    # c: System A right (1), System B wrong (0)
    b = int(np.sum((ca == 0) & (cb == 1)))
    c = int(np.sum((ca == 1) & (cb == 0)))

    discordant_total = b + c
    if discordant_total == 0:
        return {"statistic": 0.0, "p_value": 1.0, "b_count": 0.0, "c_count": 0.0}

    # Use exact binomial test if discordant count is small (< 25)
    if discordant_total < 25:
        res = stats.binomtest(k=min(b, c), n=discordant_total, p=0.5, alternative="two-sided")
        stat = float((abs(b - c) - 1.0) ** 2 / discordant_total)
        pval = float(res.pvalue)
    else:
        # Edwards' continuity-corrected Chi-Square: (|b - c| - 1)^2 / (b + c)
        stat = float((abs(b - c) - 1.0) ** 2 / discordant_total)
        pval = float(stats.chi2.sf(stat, df=1))

    return {
        "statistic": round(stat, 4),
        "p_value": round(pval, 5),
        "b_count": float(b),
        "c_count": float(c),
    }


def bonferroni_correction(
    p_values: Sequence[float], alpha: float = 0.05
) -> list[bool]:
    """Apply Bonferroni FWER family-wise error rate correction to p-values.

    Args:
        p_values: Sequence of computed p-values.
        alpha: Target significance threshold.

    Returns:
        List of booleans where True marks statistically significant rejection of H0.
    """
    m = len(p_values)
    if m == 0:
        return []

    adjusted_alpha = alpha / m
    return [bool(p <= adjusted_alpha) for p in p_values]


def compute_confidence_intervals_for_metrics(
    metrics_dict: dict[str, float],
    results_list: Sequence[PipelineResult],
    ci: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Compute non-parametric bootstrap confidence intervals for all evaluated metrics.

    Args:
        metrics_dict: Point estimates of evaluated metrics.
        results_list: Underlying pipeline execution results.
        ci: Confidence level (e.g. 0.95).
        n_bootstrap: Number of bootstrap iterations.
        seed: Random seed.

    Returns:
        Dictionary mapping metric names to {'point': ..., 'ci_lower': ..., 'ci_upper': ...}.
    """
    from src.evaluation.grounding_metrics import (
        compute_claim_support_rate,
        compute_contradiction_rate,
        compute_unsupported_claim_rate,
    )

    metric_fns: dict[str, Callable[[Sequence[PipelineResult]], float]] = {
        "claim_support_rate": compute_claim_support_rate,
        "unsupported_claim_rate": compute_unsupported_claim_rate,
        "contradiction_rate": compute_contradiction_rate,
    }

    ci_results: dict[str, dict[str, float]] = {}

    for name, point_val in metrics_dict.items():
        fn = metric_fns.get(name)
        if fn is not None and results_list:
            _, low, high = bootstrap_confidence_interval(
                results_list, fn, n_bootstrap=n_bootstrap, ci=ci, seed=seed
            )
            ci_results[name] = {
                "point": round(point_val, 4),
                "ci_lower": round(low, 4),
                "ci_upper": round(high, 4),
            }
        else:
            # Fallback when sample-level metric function is not directly available
            ci_results[name] = {
                "point": round(point_val, 4),
                "ci_lower": round(point_val, 4),
                "ci_upper": round(point_val, 4),
            }

    return ci_results
