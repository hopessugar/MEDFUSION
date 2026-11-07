"""Citation validation for medical answer verification."""
from __future__ import annotations

import logging
import re
from typing import Optional

from pydantic import BaseModel, Field

from src.models.schemas import (
    AtomicClaim,
    Citation,
    ClaimSupportStatus,
    RetrievedPassage,
)

logger = logging.getLogger(__name__)


class CitationValidationConfig(BaseModel):
    """Configuration for citation validation."""
    strict_mode: bool = True
    check_passage_existence: bool = True
    check_content_relevance: bool = True
    relevance_keyword_overlap_threshold: float = 0.3
    min_keyword_length: int = 3


class CitationValidationResult(BaseModel):
    """Result of citation validation for a single claim."""
    claim_id: str
    claim_text: str
    cited_passage_ids: list[str] = Field(default_factory=list)
    valid_citations: list[str] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    missing_passage_ids: list[str] = Field(default_factory=list)
    relevance_scores: dict[str, float] = Field(default_factory=dict)
    is_valid: bool = False
    reason: str = ""


class OverallCitationResult(BaseModel):
    """Overall citation validation results for an answer."""
    claim_results: list[CitationValidationResult] = Field(default_factory=list)
    total_citations: int = 0
    valid_citations: int = 0
    invalid_citations: int = 0
    missing_citations: int = 0
    citation_precision: float = 0.0
    citation_recall: float = 0.0
    overall_valid: bool = False


class CitationValidator:
    """Validates that citations in generated answers reference real passages
    and that the cited passages are relevant to the claims made.
    
    This provides deterministic, non-LLM citation checking to complement
    the LLM-based verification from the Qwen verifier.
    """

    # Common medical stopwords to exclude from keyword matching
    MEDICAL_STOPWORDS = frozenset({
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'can', 'shall',
        'of', 'in', 'to', 'for', 'with', 'on', 'at', 'by', 'from',
        'as', 'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'between', 'and', 'but', 'or', 'not', 'no', 'nor',
        'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them',
        'their', 'we', 'our', 'you', 'your', 'he', 'she', 'his', 'her',
        'patient', 'patients', 'study', 'studies', 'treatment', 'results',
        'showed', 'found', 'also', 'however', 'although', 'therefore',
        'which', 'who', 'whom', 'what', 'when', 'where', 'how', 'than',
        'both', 'each', 'all', 'some', 'any', 'most', 'other', 'such',
    })

    def __init__(self, config: Optional[CitationValidationConfig] = None) -> None:
        self.config = config or CitationValidationConfig()

    def validate(
        self,
        claims: list[AtomicClaim],
        passages: list[RetrievedPassage],
    ) -> OverallCitationResult:
        """Validate all citations across all claims.

        Args:
            claims: List of atomic claims with citation references.
            passages: List of retrieved passages available as evidence.

        Returns:
            Overall citation validation result with per-claim details.
        """
        # Build passage lookup
        passage_map: dict[str, RetrievedPassage] = {}
        for p in passages:
            passage_map[p.passage_id] = p
            # Also index by common alias patterns
            # e.g., "passage_0001" should match "P1" references
            match = re.match(r'.*?(\d+)', p.passage_id)
            if match:
                num = int(match.group(1))
                alias = f"passage_{num:04d}"
                if alias not in passage_map:
                    passage_map[alias] = p

        claim_results: list[CitationValidationResult] = []
        total_citations = 0
        valid_count = 0
        invalid_count = 0
        missing_count = 0

        for claim in claims:
            result = self._validate_claim_citations(claim, passage_map)
            claim_results.append(result)
            
            total_citations += len(result.cited_passage_ids)
            valid_count += len(result.valid_citations)
            invalid_count += len(result.invalid_citations)
            missing_count += len(result.missing_passage_ids)

        # Calculate overall metrics
        citation_precision = (
            valid_count / total_citations if total_citations > 0 else 0.0
        )
        
        # Citation recall: proportion of claims that have at least one valid citation
        claims_with_citations = sum(
            1 for r in claim_results if len(r.cited_passage_ids) > 0
        )
        claims_with_valid = sum(
            1 for r in claim_results if len(r.valid_citations) > 0
        )
        citation_recall = (
            claims_with_valid / len(claims) if len(claims) > 0 else 0.0
        )

        overall_valid = (
            invalid_count == 0 and missing_count == 0
            if self.config.strict_mode
            else citation_precision >= 0.5
        )

        result = OverallCitationResult(
            claim_results=claim_results,
            total_citations=total_citations,
            valid_citations=valid_count,
            invalid_citations=invalid_count,
            missing_citations=missing_count,
            citation_precision=citation_precision,
            citation_recall=citation_recall,
            overall_valid=overall_valid,
        )

        logger.info(
            "Citation validation: %d/%d valid (precision=%.3f, recall=%.3f)",
            valid_count,
            total_citations,
            citation_precision,
            citation_recall,
        )
        return result

    def _validate_claim_citations(
        self,
        claim: AtomicClaim,
        passage_map: dict[str, RetrievedPassage],
    ) -> CitationValidationResult:
        """Validate citations for a single claim."""
        cited_ids = [c.passage_id for c in claim.citations]
        valid: list[str] = []
        invalid: list[str] = []
        missing: list[str] = []
        relevance_scores: dict[str, float] = {}

        for pid in cited_ids:
            if pid not in passage_map:
                if self.config.check_passage_existence:
                    missing.append(pid)
                    logger.debug(
                        "Citation %s references non-existent passage", pid
                    )
                continue

            passage = passage_map[pid]

            if self.config.check_content_relevance:
                score = self._compute_keyword_relevance(
                    claim.claim_text, passage.text
                )
                relevance_scores[pid] = score

                if score >= self.config.relevance_keyword_overlap_threshold:
                    valid.append(pid)
                else:
                    invalid.append(pid)
                    logger.debug(
                        "Citation %s has low relevance (%.3f) to claim: %s",
                        pid,
                        score,
                        claim.claim_text[:80],
                    )
            else:
                # If not checking relevance, existence is sufficient
                valid.append(pid)

        is_valid = len(invalid) == 0 and len(missing) == 0
        reason = ""
        if missing:
            reason += f"Missing passages: {missing}. "
        if invalid:
            reason += f"Irrelevant citations: {invalid}. "
        if not cited_ids:
            reason = "No citations provided."
            is_valid = False

        return CitationValidationResult(
            claim_id=claim.claim_id,
            claim_text=claim.claim_text,
            cited_passage_ids=cited_ids,
            valid_citations=valid,
            invalid_citations=invalid,
            missing_passage_ids=missing,
            relevance_scores=relevance_scores,
            is_valid=is_valid,
            reason=reason,
        )

    def _compute_keyword_relevance(
        self,
        claim_text: str,
        passage_text: str,
    ) -> float:
        """Compute keyword overlap relevance between claim and passage.

        Uses Jaccard-like coefficient on content keywords, excluding
        common stopwords and very short tokens.
        """
        claim_keywords = self._extract_keywords(claim_text)
        passage_keywords = self._extract_keywords(passage_text)

        if not claim_keywords:
            return 0.0

        overlap = claim_keywords & passage_keywords
        # Asymmetric: what fraction of claim keywords appear in passage
        return len(overlap) / len(claim_keywords)

    def _extract_keywords(self, text: str) -> set[str]:
        """Extract content keywords from text."""
        # Tokenize: lowercase, split on non-alphanumeric
        tokens = re.findall(r'[a-z0-9]+', text.lower())
        # Filter stopwords and short tokens
        keywords = {
            t for t in tokens
            if t not in self.MEDICAL_STOPWORDS
            and len(t) >= self.config.min_keyword_length
        }
        return keywords

