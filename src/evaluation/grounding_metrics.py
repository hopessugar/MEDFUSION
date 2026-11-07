"""Grounding and citation evaluation metrics for MedFusion-RAFT.

Measures atomic claim verification rates, citation precision and recall,
and factual contradiction ratios against retrieved clinical evidence.
"""

from __future__ import annotations

from typing import Sequence

from src.models.schemas import ClaimSupportStatus, PipelineResult
from src.utils.logging import get_logger

logger = get_logger(__name__)


def compute_claim_support_rate(results: Sequence[PipelineResult]) -> float:
    """Compute the overall fraction of atomic claims verified as supported.

    Args:
        results: List of execution results with verification metadata.

    Returns:
        Fraction of claims supported in [0.0, 1.0].
    """
    total_claims = 0
    supported_claims = 0

    for r in results:
        ver = r.verification_result
        if ver is None:
            continue
        total_claims += (
            ver.supported_claims + ver.unsupported_claims + ver.contradicted_claims
        )
        supported_claims += ver.supported_claims

    if total_claims == 0:
        return 0.0
    return supported_claims / total_claims


def compute_unsupported_claim_rate(results: Sequence[PipelineResult]) -> float:
    """Compute the overall fraction of atomic claims lacking evidence support.

    Args:
        results: List of execution results with verification metadata.

    Returns:
        Fraction of claims unsupported in [0.0, 1.0].
    """
    total_claims = 0
    unsupported_claims = 0

    for r in results:
        ver = r.verification_result
        if ver is None:
            continue
        total_claims += (
            ver.supported_claims + ver.unsupported_claims + ver.contradicted_claims
        )
        unsupported_claims += ver.unsupported_claims

    if total_claims == 0:
        return 0.0
    return unsupported_claims / total_claims


def compute_contradiction_rate(results: Sequence[PipelineResult]) -> float:
    """Compute the fraction of atomic claims contradicted by evidence.

    Args:
        results: List of execution results with verification metadata.

    Returns:
        Fraction of claims contradicted in [0.0, 1.0].
    """
    total_claims = 0
    contradicted_claims = 0

    for r in results:
        ver = r.verification_result
        if ver is None:
            continue
        total_claims += (
            ver.supported_claims + ver.unsupported_claims + ver.contradicted_claims
        )
        contradicted_claims += ver.contradicted_claims

    if total_claims == 0:
        return 0.0
    return contradicted_claims / total_claims


def compute_citation_precision(results: Sequence[PipelineResult]) -> float:
    """Compute citation precision: fraction of cited passages that actually exist and support claims.

    Args:
        results: List of execution results.

    Returns:
        Citation precision score in [0.0, 1.0].
    """
    total_citations = 0
    valid_citations = 0

    for r in results:
        ver = r.verification_result
        if ver is None:
            continue

        retrieved_ids = {p.passage_id for p in r.retrieved_passages}
        for claim in ver.claim_results:
            for cit in claim.citations:
                total_citations += 1
                # Valid citation if passage exists in retrieved context and claim is supported
                if cit.passage_id in retrieved_ids and claim.support_status in (
                    ClaimSupportStatus.SUPPORTED,
                    ClaimSupportStatus.PARTIALLY_SUPPORTED,
                ):
                    valid_citations += 1

    if total_citations == 0:
        return 1.0 if not any(r.retrieved_passages for r in results) else 0.0
    return valid_citations / total_citations


def compute_citation_recall(results: Sequence[PipelineResult]) -> float:
    """Compute citation recall: fraction of supported atomic claims that include valid citations.

    Args:
        results: List of execution results.

    Returns:
        Citation recall score in [0.0, 1.0].
    """
    supported_claims_count = 0
    cited_supported_claims = 0

    for r in results:
        ver = r.verification_result
        if ver is None:
            continue

        for claim in ver.claim_results:
            if claim.support_status in (
                ClaimSupportStatus.SUPPORTED,
                ClaimSupportStatus.PARTIALLY_SUPPORTED,
            ):
                supported_claims_count += 1
                if claim.citations:
                    cited_supported_claims += 1

    if supported_claims_count == 0:
        return 0.0
    return cited_supported_claims / supported_claims_count


def evaluate_grounding(results: list[PipelineResult]) -> dict[str, float]:
    """Calculate aggregated evidence grounding and factual attribution metrics.

    Args:
        results: Execution results containing claim-level verification records.

    Returns:
        Dictionary of calculated grounding metrics.
    """
    support_rate = compute_claim_support_rate(results)
    unsupported_rate = compute_unsupported_claim_rate(results)
    contradiction_rate = compute_contradiction_rate(results)
    citation_precision = compute_citation_precision(results)
    citation_recall = compute_citation_recall(results)

    metrics = {
        "claim_support_rate": round(support_rate, 4),
        "unsupported_claim_rate": round(unsupported_rate, 4),
        "contradiction_rate": round(contradiction_rate, 4),
        "citation_precision": round(citation_precision, 4),
        "citation_recall": round(citation_recall, 4),
    }

    logger.info("Grounding evaluation completed over %d results: %s", len(results), metrics)
    return metrics
