"""Entailment checking using NLI models for evidence verification."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from pydantic import BaseModel

from src.models.schemas import AtomicClaim, ClaimSupportStatus, RetrievedPassage

logger = logging.getLogger(__name__)


class EntailmentConfig(BaseModel):
    """Configuration for entailment checking."""
    model_id: str = "cross-encoder/nli-deberta-v3-base"
    threshold_supported: float = 0.7
    threshold_contradicted: float = 0.3
    batch_size: int = 16
    device: str = "cpu"
    max_length: int = 512


class EntailmentResult(BaseModel):
    """Result of entailment check for a single claim-passage pair."""
    claim_text: str
    passage_id: str
    entailment_score: float  # P(entailment)
    contradiction_score: float  # P(contradiction)
    neutral_score: float  # P(neutral)
    status: ClaimSupportStatus


class EntailmentChecker:
    """Checks whether evidence passages entail, contradict, or are
    neutral toward atomic claims using a pre-trained NLI model.

    This provides deterministic, model-based evidence support checking
    that complements the LLM-based Qwen verification.
    """

    # NLI label mapping for cross-encoder/nli-deberta-v3-base
    # The model outputs logits for [contradiction, entailment, neutral]
    LABEL_CONTRADICTION = 0
    LABEL_ENTAILMENT = 1
    LABEL_NEUTRAL = 2

    def __init__(self, config: Optional[EntailmentConfig] = None) -> None:
        self.config = config or EntailmentConfig()
        self._model = None
        self._tokenizer = None
        self._loaded = False

    def load(self) -> None:
        """Load the NLI model and tokenizer."""
        if self._loaded:
            return

        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )

            logger.info("Loading NLI model: %s", self.config.model_id)
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_id
            )
            
            if self.config.device != "cpu":
                try:
                    import torch
                    self._model = self._model.to(self.config.device)
                except Exception:
                    logger.warning("Failed to move NLI model to %s, using CPU", self.config.device)
            
            self._model.eval()
            self._loaded = True
            logger.info("NLI model loaded successfully")
        except ImportError:
            logger.error(
                "transformers not installed. Install with: pip install transformers"
            )
            raise
        except Exception as e:
            logger.error("Failed to load NLI model: %s", e)
            raise

    def unload(self) -> None:
        """Unload the model from memory."""
        self._model = None
        self._tokenizer = None
        self._loaded = False
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def check_entailment(
        self,
        claim: str,
        passage: str,
    ) -> EntailmentResult:
        """Check if a passage entails a claim.

        Args:
            claim: The atomic claim text (hypothesis).
            passage: The evidence passage text (premise).

        Returns:
            EntailmentResult with scores and classification.
        """
        if not self._loaded:
            self.load()

        import torch

        # Tokenize the premise-hypothesis pair
        inputs = self._tokenizer(
            passage,
            claim,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
            padding=True,
        )

        if self.config.device != "cpu" and self._model is not None:
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        contradiction_score = float(probs[self.LABEL_CONTRADICTION])
        entailment_score = float(probs[self.LABEL_ENTAILMENT])
        neutral_score = float(probs[self.LABEL_NEUTRAL])

        # Classify based on thresholds
        if entailment_score >= self.config.threshold_supported:
            status = ClaimSupportStatus.SUPPORTED
        elif contradiction_score >= self.config.threshold_contradicted:
            status = ClaimSupportStatus.CONTRADICTED
        elif entailment_score >= 0.4:
            status = ClaimSupportStatus.PARTIALLY_SUPPORTED
        else:
            status = ClaimSupportStatus.INSUFFICIENT_EVIDENCE

        return EntailmentResult(
            claim_text=claim,
            passage_id="",  # Set by caller
            entailment_score=entailment_score,
            contradiction_score=contradiction_score,
            neutral_score=neutral_score,
            status=status,
        )

    def check_claim_against_passages(
        self,
        claim: AtomicClaim,
        passages: list[RetrievedPassage],
    ) -> tuple[ClaimSupportStatus, float, list[EntailmentResult]]:
        """Check a claim against multiple passages and return aggregate result.

        Args:
            claim: The atomic claim to verify.
            passages: List of candidate evidence passages.

        Returns:
            Tuple of (overall_status, best_score, individual_results).
        """
        if not passages:
            return ClaimSupportStatus.INSUFFICIENT_EVIDENCE, 0.0, []

        results: list[EntailmentResult] = []
        best_entailment = 0.0
        worst_contradiction = 0.0

        for passage in passages:
            result = self.check_entailment(claim.claim_text, passage.text)
            result.passage_id = passage.passage_id
            results.append(result)

            best_entailment = max(best_entailment, result.entailment_score)
            worst_contradiction = max(
                worst_contradiction, result.contradiction_score
            )

        # Determine overall status based on best evidence
        if worst_contradiction >= self.config.threshold_contradicted and best_entailment < self.config.threshold_supported:
            overall_status = ClaimSupportStatus.CONTRADICTED
        elif best_entailment >= self.config.threshold_supported:
            overall_status = ClaimSupportStatus.SUPPORTED
        elif best_entailment >= 0.4:
            overall_status = ClaimSupportStatus.PARTIALLY_SUPPORTED
        else:
            overall_status = ClaimSupportStatus.INSUFFICIENT_EVIDENCE

        return overall_status, best_entailment, results


class MockEntailmentChecker(EntailmentChecker):
    """Mock entailment checker for CPU-only testing without model downloads."""

    def __init__(self, config: Optional[EntailmentConfig] = None) -> None:
        super().__init__(config)
        self._loaded = True  # Always "loaded"

    def load(self) -> None:
        """No-op for mock."""
        self._loaded = True

    def check_entailment(
        self,
        claim: str,
        passage: str,
    ) -> EntailmentResult:
        """Return mock entailment results based on keyword overlap."""
        # Simple heuristic: keyword overlap ratio
        claim_words = set(claim.lower().split())
        passage_words = set(passage.lower().split())
        
        if not claim_words:
            overlap = 0.0
        else:
            overlap = len(claim_words & passage_words) / len(claim_words)

        if overlap > 0.5:
            entailment = 0.8
            contradiction = 0.05
            neutral = 0.15
            status = ClaimSupportStatus.SUPPORTED
        elif overlap > 0.3:
            entailment = 0.5
            contradiction = 0.1
            neutral = 0.4
            status = ClaimSupportStatus.PARTIALLY_SUPPORTED
        else:
            entailment = 0.15
            contradiction = 0.1
            neutral = 0.75
            status = ClaimSupportStatus.INSUFFICIENT_EVIDENCE

        return EntailmentResult(
            claim_text=claim,
            passage_id="",
            entailment_score=entailment,
            contradiction_score=contradiction,
            neutral_score=neutral,
            status=status,
        )

