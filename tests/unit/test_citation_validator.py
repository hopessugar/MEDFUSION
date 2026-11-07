"""Unit tests for citation validation against retrieved evidence passages."""

from __future__ import annotations

import pytest

from src.models.schemas import (
    AtomicClaim,
    Citation,
    RetrievedPassage,
)
from src.verification.citation_validator import (
    CitationValidationConfig,
    CitationValidator,
)


def test_citation_validator_valid_citation() -> None:
    """Test correctly cited passage with keyword relevance."""
    passages = [
        RetrievedPassage(
            passage_id="p1",
            doc_id="d1",
            text="Metformin reduces hepatic gluconeogenesis and is the first-line medication for type 2 diabetes.",
            score=0.95,
        )
    ]
    cit = Citation(claim_text="Metformin decreases hepatic gluconeogenesis.", passage_id="p1")
    claim = AtomicClaim(
        claim_id="c1",
        claim_text="Metformin decreases hepatic gluconeogenesis.",
        citations=[cit],
    )

    validator = CitationValidator(config=CitationValidationConfig(relevance_keyword_overlap_threshold=0.2))
    res = validator.validate([claim], passages)

    assert res.total_citations == 1
    assert res.valid_citations == 1
    assert res.invalid_citations == 0
    assert res.missing_citations == 0
    assert res.citation_precision == 1.0
    assert res.overall_valid is True


def test_citation_validator_missing_passage() -> None:
    """Test hallucinated passage citation that does not exist in retrieved results."""
    passages = [
        RetrievedPassage(
            passage_id="p1",
            doc_id="d1",
            text="Lisinopril is an ACE inhibitor used in heart failure management.",
            score=0.90,
        )
    ]
    # Claim cites p99 which is not in passages
    cit = Citation(claim_text="Lisinopril improves ejection fraction.", passage_id="p99")
    claim = AtomicClaim(
        claim_id="c1",
        claim_text="Lisinopril improves ejection fraction.",
        citations=[cit],
    )

    validator = CitationValidator()
    res = validator.validate([claim], passages)

    assert res.total_citations == 1
    assert res.valid_citations == 0
    assert res.missing_citations == 1
    assert res.overall_valid is False
    assert len(res.claim_results[0].missing_passage_ids) == 1


def test_citation_validator_irrelevant_passage() -> None:
    """Test existing passage citation that lacks semantic keyword relevance to the claim."""
    passages = [
        RetrievedPassage(
            passage_id="p1",
            doc_id="d1",
            text="Cataract surgery involves phacoemulsification and intraocular lens implantation.",
            score=0.50,
        )
    ]
    # Claim is about myocardial infarction, cited passage is ophthalmology
    cit = Citation(claim_text="Aspirin reduces coronary artery stent thrombosis.", passage_id="p1")
    claim = AtomicClaim(
        claim_id="c1",
        claim_text="Aspirin reduces coronary artery stent thrombosis.",
        citations=[cit],
    )

    validator = CitationValidator(
        config=CitationValidationConfig(
            check_content_relevance=True,
            relevance_keyword_overlap_threshold=0.4,
        )
    )
    res = validator.validate([claim], passages)

    assert res.total_citations == 1
    assert res.valid_citations == 0
    assert res.invalid_citations == 1
    assert res.overall_valid is False


def test_citation_validator_empty_citations() -> None:
    """Test handling claims that do not contain any citations."""
    passages = [
        RetrievedPassage(
            passage_id="p1",
            doc_id="d1",
            text="Antibiotic therapy should be tailored to culture sensitivities.",
            score=0.88,
        )
    ]
    claim = AtomicClaim(
        claim_id="c1",
        claim_text="Antibiotic stewardship is encouraged.",
        citations=[],
    )

    validator = CitationValidator()
    res = validator.validate([claim], passages)

    assert res.total_citations == 0
    assert res.valid_citations == 0
    assert res.overall_valid is True
