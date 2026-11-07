"""Claim extraction from generated medical answers."""
from __future__ import annotations

import re
import logging
from typing import Optional

from pydantic import BaseModel, Field

from src.models.schemas import AtomicClaim, Citation, ClaimSupportStatus

logger = logging.getLogger(__name__)


class ClaimExtractionConfig(BaseModel):
    """Configuration for claim extraction."""
    max_claims_per_answer: int = 20
    min_claim_length: int = 10
    split_on_conjunctions: bool = True
    extract_citations: bool = True


class ClaimExtractor:
    """Extracts atomic factual claims from generated answers.
    
    Splits complex answer text into individual verifiable claims,
    each with associated citation references if present.
    """

    # Patterns for citation references like [P1], [P2], [Passage 3]
    CITATION_PATTERN = re.compile(
        r'\[(?:P|Passage\s?)(\d+)\]',
        re.IGNORECASE,
    )
    
    # Sentence splitting pattern (handles sentence terminal punctuation)
    SENTENCE_PATTERN = re.compile(
        r'(?<=[.!?])\s+(?=[A-Z])',
    )
    ABBREVIATIONS = (
        "dr.", "mr.", "mrs.", "ms.", "prof.", "vs.", "e.g.", "i.e.", "etc.",
        "dr", "mr", "mrs", "ms", "prof", "vs", "e.g", "i.e", "etc"
    )

    def __init__(self, config: Optional[ClaimExtractionConfig] = None) -> None:
        self.config = config or ClaimExtractionConfig()

    def extract_claims(self, answer_text: str) -> list[AtomicClaim]:
        """Extract atomic claims from answer text.

        Args:
            answer_text: The generated answer text to decompose.

        Returns:
            List of AtomicClaim objects, each representing one
            verifiable factual statement with associated citations.
        """
        if not answer_text or not answer_text.strip():
            return []

        # Clean the answer text
        cleaned = self._clean_text(answer_text)
        
        # Split into sentences
        sentences = self._split_sentences(cleaned)
        
        # Further decompose compound sentences if configured
        if self.config.split_on_conjunctions:
            sentences = self._split_compound_sentences(sentences)
        
        # Filter and create claims
        claims: list[AtomicClaim] = []
        for idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            
            if len(sentence) < self.config.min_claim_length:
                continue
            
            if len(claims) >= self.config.max_claims_per_answer:
                logger.warning(
                    "Reached max claims limit (%d), truncating",
                    self.config.max_claims_per_answer,
                )
                break
            
            # Skip non-factual sentences (questions, meta-commentary)
            if self._is_non_factual(sentence):
                continue
            
            # Extract citations from this sentence
            citations = self._extract_citations(sentence) if self.config.extract_citations else []
            
            # Remove citation markers from claim text for cleaner representation
            claim_text = self.CITATION_PATTERN.sub('', sentence).strip()
            claim_text = re.sub(r'\s{2,}', ' ', claim_text)
            
            if len(claim_text) < self.config.min_claim_length:
                continue

            claim = AtomicClaim(
                claim_id=f"claim_{idx:03d}",
                claim_text=claim_text,
                citations=citations,
                support_status=None,  # Will be set during verification
                entailment_score=None,
            )
            claims.append(claim)

        logger.info("Extracted %d atomic claims from answer", len(claims))
        return claims

    def _clean_text(self, text: str) -> str:
        """Remove formatting artifacts and normalize whitespace."""
        # Remove markdown formatting
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        # Remove bullet points and list markers
        text = re.sub(r'^\s*[-•*]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
        # Normalize whitespace
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip()

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into individual sentences."""
        raw_parts = self.SENTENCE_PATTERN.split(text)
        sentences: list[str] = []
        for part in raw_parts:
            part_s = part.strip()
            if not part_s:
                continue
            if sentences and any(sentences[-1].lower().rstrip('.').endswith(abbr.rstrip('.')) for abbr in self.ABBREVIATIONS if sentences[-1].lower().endswith(abbr) or sentences[-1].lower().endswith(abbr + '.')):
                sentences[-1] = f"{sentences[-1]} {part_s}"
            else:
                sentences.append(part_s)
        return sentences

    def _split_compound_sentences(self, sentences: list[str]) -> list[str]:
        """Split compound sentences on conjunctions where safe.

        Only splits on 'and', 'but', 'however', 'additionally',
        'furthermore', 'moreover' when both parts form independent
        clauses (heuristic: both sides are long enough).
        """
        result: list[str] = []
        conjunction_pattern = re.compile(
            r',\s*(?:and|but|however|additionally|furthermore|moreover)\s+',
            re.IGNORECASE,
        )
        
        for sentence in sentences:
            parts = conjunction_pattern.split(sentence)
            if len(parts) > 1:
                for part in parts:
                    part = part.strip().rstrip(',').strip()
                    if len(part) >= self.config.min_claim_length:
                        result.append(part)
                    else:
                        # Too short to stand alone; merge back
                        if result:
                            result[-1] = result[-1] + ', ' + part
                        else:
                            result.append(part)
            else:
                result.append(sentence)
        
        return result

    def _is_non_factual(self, sentence: str) -> bool:
        """Check if a sentence is non-factual (question, hedging, meta)."""
        lower = sentence.lower().strip()
        
        # Questions
        if lower.endswith('?'):
            return True
        
        # Pure meta-commentary
        meta_prefixes = [
            'based on the provided evidence',
            'according to the passages',
            'the evidence suggests',
            'in summary',
            'to summarize',
            'in conclusion',
            'it should be noted that',
            'it is important to note',
            'as mentioned above',
            'as stated earlier',
        ]
        for prefix in meta_prefixes:
            if lower.startswith(prefix):
                # Keep these — they may contain factual claims after the prefix
                return False
        
        # Pure hedging without substance
        pure_hedge = [
            'more research is needed',
            'further studies are required',
            'this is a complex topic',
            'consult a healthcare professional',
            'please see a doctor',
        ]
        for hedge in pure_hedge:
            if lower == hedge or lower.rstrip('.') == hedge:
                return True
        
        return False

    def _extract_citations(self, sentence: str) -> list[Citation]:
        """Extract citation references from a sentence."""
        citations: list[Citation] = []
        seen_ids: set[str] = set()
        
        for match in self.CITATION_PATTERN.finditer(sentence):
            passage_num = match.group(1)
            passage_id = f"passage_{int(passage_num):04d}"
            
            if passage_id in seen_ids:
                continue
            seen_ids.add(passage_id)
            
            # Get the claim text (sentence without citations)
            claim_text = self.CITATION_PATTERN.sub('', sentence).strip()
            
            citation = Citation(
                claim_text=claim_text,
                passage_id=passage_id,
                relevance_score=None,
            )
            citations.append(citation)
        
        return citations

