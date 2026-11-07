"""Unit tests for atomic claim extraction and citation parsing."""

from __future__ import annotations

import pytest

from src.verification.claim_extractor import (
    ClaimExtractionConfig,
    ClaimExtractor,
)


def test_claim_extractor_basic_extraction() -> None:
    """Test standard sentence splitting and claim creation."""
    text = (
        "Metformin is the initial drug of choice for type 2 diabetes [P1]. "
        "It acts primarily by decreasing hepatic glucose production [P2]. "
        "Patients should monitor renal function annually."
    )
    extractor = ClaimExtractor()
    claims = extractor.extract_claims(text)

    assert len(claims) >= 3
    assert claims[0].claim_id == "claim_000"
    assert "Metformin is the initial drug of choice" in claims[0].claim_text
    assert len(claims[0].citations) == 1
    assert claims[0].citations[0].passage_id == "passage_0001"
    assert len(claims[1].citations) == 1
    assert claims[1].citations[0].passage_id == "passage_0002"
    assert len(claims[2].citations) == 0


def test_claim_extractor_compound_splitting() -> None:
    """Test decomposing compound sentences on conjunctions."""
    text = (
        "ACE inhibitors reduce systemic vascular resistance, and they preserve renal hemodynamics. "
        "However, serum potassium must be monitored closely."
    )
    cfg = ClaimExtractionConfig(split_on_conjunctions=True, min_claim_length=10)
    extractor = ClaimExtractor(config=cfg)
    claims = extractor.extract_claims(text)

    # Compound sentence should produce split atomic propositions
    assert len(claims) >= 2
    claim_texts = [c.claim_text.lower() for c in claims]
    assert any("reduce systemic vascular resistance" in ct for ct in claim_texts)


def test_claim_extractor_non_factual_filtering() -> None:
    """Test filtering out questions and non-factual introductory statements."""
    text = (
        "Should beta-blockers be used as first line therapy? "
        "Consult a healthcare professional. "
        "Thiazide diuretics are recommended for stage 1 hypertension [P1]."
    )
    extractor = ClaimExtractor()
    claims = extractor.extract_claims(text)

    # Questions and meta-commentary should be skipped; only the factual assertion remains
    assert len(claims) == 1
    assert "Thiazide diuretics" in claims[0].claim_text


def test_claim_extractor_citation_variants() -> None:
    """Test extracting diverse citation patterns like [P1], [P2], [Passage 3]."""
    text = (
        "Aspirin reduces myocardial infarction recurrence [P1]. "
        "Statins stabilize vulnerable plaques [Passage 2]. "
        "Dual antiplatelet therapy is indicated post-stenting [P03]."
    )
    extractor = ClaimExtractor()
    claims = extractor.extract_claims(text)

    assert len(claims) == 3
    assert claims[0].citations[0].passage_id == "passage_0001"
    assert claims[1].citations[0].passage_id == "passage_0002"
    assert claims[2].citations[0].passage_id == "passage_0003"


def test_claim_extractor_empty_and_short() -> None:
    """Test edge cases with empty or ultra-short input."""
    extractor = ClaimExtractor()
    assert extractor.extract_claims("") == []
    assert extractor.extract_claims("   \n\t") == []
    assert extractor.extract_claims("Yes.") == []
