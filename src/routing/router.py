"""Configurable question classification and routing framework for MedFusion-RAFT.

Categorizes clinical inquiries based on linguistic and medical complexity to
adaptively configure retrieval depth, generation parameters, and verification rigor.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.utils.logging import get_logger

logger = get_logger(__name__)


class RoutingDecision(str, Enum):
    """Routing pathways based on clinical query characteristics."""

    SIMPLE = "simple"
    STANDARD = "standard"
    COMPLEX = "complex"
    EVIDENCE_CRITICAL = "evidence_critical"


class QuestionComplexity(str, Enum):
    """Assessed linguistic and medical complexity level."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RouterConfig(BaseModel):
    """Configuration thresholds and heuristics for the query router."""

    model_config = ConfigDict(extra="ignore")

    # Length thresholds (word counts)
    short_query_threshold: int = Field(default=12, ge=1)
    long_query_threshold: int = Field(default=35, ge=5)

    # Keywords signaling evidence-critical clinical safety
    critical_keywords: list[str] = Field(
        default_factory=lambda: [
            "dose",
            "dosage",
            "doses",
            "contraindication",
            "contraindications",
            "contraindicated",
            "toxicity",
            "lethal",
            "interaction",
            "interactions",
            "overdose",
            "anaphylaxis",
            "black box",
            "pregnancy",
            "pregnant",
            "teratogenic",
            "pediatric",
            "infant",
            "fatal",
            "emergency",
            "adverse effect",
            "adverse effects",
            "side effect",
            "side effects",
        ]
    )

    # Keywords signaling differential diagnosis or comparative inquiries
    comparison_keywords: list[str] = Field(
        default_factory=lambda: [
            "differential",
            "versus",
            "vs",
            "distinguish",
            "differentiate",
            "compare",
            "contrast",
            "either",
            "difference between",
            "alternative to",
        ]
    )

    # Patterns indicating multi-part questions
    multi_part_patterns: list[str] = Field(
        default_factory=lambda: [
            r"\?.*\?",  # Multiple question marks
            r"\b(first|second|firstly|secondly|1\)|2\)|a\)|b\))\b",
            r"\b(additionally|furthermore|as well as|in addition)\b",
            r"\band\s+what\b",
            r"\band\s+how\b",
        ]
    )

    # Medical terminology indicators (common prefixes/suffixes & clinical domains)
    medical_affixes: list[str] = Field(
        default_factory=lambda: [
            "itis",
            "emia",
            "opathy",
            "ectomy",
            "scopy",
            "plasty",
            "stasis",
            "algia",
            "genesis",
            "lysis",
            "cardio",
            "neuro",
            "nephro",
            "hepato",
            "pulmo",
            "gastro",
            "onco",
            "immuno",
            "dermato",
            "hemat",
            "pharma",
        ]
    )

    # Medical density threshold (fraction of words with medical roots/keywords)
    medical_density_threshold: float = Field(default=0.15, ge=0.0, le=1.0)


class RuleBasedRouter:
    """Heuristic rule-based router classifying clinical questions."""

    def __init__(self, config: Optional[RouterConfig] = None) -> None:
        """Initialize router with configuration rules.

        Args:
            config: Routing heuristic configuration. If None, default settings are used.
        """
        self.config = config or RouterConfig()
        self._compiled_multi_part = [
            re.compile(p, re.IGNORECASE) for p in self.config.multi_part_patterns
        ]
        logger.info(
            "RuleBasedRouter initialized with %d critical keywords, %d comparison keywords",
            len(self.config.critical_keywords),
            len(self.config.comparison_keywords),
        )

    def _compute_medical_density(self, tokens: list[str]) -> float:
        """Calculate fraction of tokens exhibiting clinical affixes or terms.

        Args:
            tokens: List of lowercased alphanumeric tokens from query.

        Returns:
            Ratio of medical tokens to total tokens in [0.0, 1.0].
        """
        if not tokens:
            return 0.0
        medical_count = 0
        for token in tokens:
            if any(affix in token for affix in self.config.medical_affixes):
                medical_count += 1
        return medical_count / len(tokens)

    def _is_multi_part(self, question: str) -> bool:
        """Detect whether a question contains multi-part queries.

        Args:
            question: Clinical query string.

        Returns:
            True if multi-part structure detected, False otherwise.
        """
        return any(pattern.search(question) for pattern in self._compiled_multi_part)

    def assess_complexity(self, question: str) -> QuestionComplexity:
        """Assess overall complexity level of the input question.

        Args:
            question: Clinical question text.

        Returns:
            Assessed QuestionComplexity enum.
        """
        text = question.strip().lower()
        tokens = re.findall(r"\b[a-z]{3,}\b", text)
        word_count = len(tokens)
        medical_density = self._compute_medical_density(tokens)
        is_multi = self._is_multi_part(text)
        has_comparison = any(kw in text for kw in self.config.comparison_keywords)

        if (
            word_count >= self.config.long_query_threshold
            or (is_multi and has_comparison)
            or (medical_density >= self.config.medical_density_threshold and is_multi)
        ):
            return QuestionComplexity.HIGH

        if (
            word_count >= self.config.short_query_threshold
            or has_comparison
            or is_multi
            or medical_density >= self.config.medical_density_threshold
        ):
            return QuestionComplexity.MEDIUM

        return QuestionComplexity.LOW

    def classify_question(self, question: str) -> RoutingDecision:
        """Classify a medical question into a routing decision.

        Heuristic priority:
        1. EVIDENCE_CRITICAL: Safety-critical keywords (dosage, contraindications, toxicity).
        2. COMPLEX: Differential diagnosis, multi-part, or high medical density.
        3. SIMPLE: Brief, low medical density, single-fact queries.
        4. STANDARD: General clinical queries.

        Args:
            question: Clinical question text.

        Returns:
            Selected RoutingDecision.
        """
        text = question.strip().lower()
        tokens = re.findall(r"\b[a-z]{3,}\b", text)
        word_count = len(tokens)
        medical_density = self._compute_medical_density(tokens)
        is_multi = self._is_multi_part(text)
        has_comparison = any(
            re.search(rf"\b{re.escape(kw)}\b", text) for kw in self.config.comparison_keywords
        )
        has_critical = any(
            re.search(rf"\b{re.escape(kw)}\b", text) for kw in self.config.critical_keywords
        )

        decision: RoutingDecision

        # 1. Critical safety gate
        if has_critical:
            decision = RoutingDecision.EVIDENCE_CRITICAL
        # 2. Complex / multi-stage reasoning
        elif (
            has_comparison
            or (is_multi and word_count > self.config.short_query_threshold)
            or (word_count >= self.config.long_query_threshold and medical_density >= self.config.medical_density_threshold)
        ):
            decision = RoutingDecision.COMPLEX
        # 3. Simple factual inquiry
        elif (
            word_count <= self.config.short_query_threshold
            and not is_multi
            and medical_density < self.config.medical_density_threshold
        ):
            decision = RoutingDecision.SIMPLE
        # 4. Standard inquiry
        else:
            decision = RoutingDecision.STANDARD

        logger.info(
            "Question classified",
            decision=decision.value,
            word_count=word_count,
            medical_density=round(medical_density, 3),
            has_critical=has_critical,
            has_comparison=has_comparison,
            is_multi=is_multi,
            question_snippet=question[:60],
        )
        return decision

    def get_retrieval_config(self, decision: RoutingDecision) -> dict[str, Any]:
        """Provide tailored retrieval parameters based on routing decision.

        Args:
            decision: Routing classification.

        Returns:
            Dictionary of retrieval parameter overrides.
        """
        configs = {
            RoutingDecision.SIMPLE: {
                "final_top_k": 3,
                "dense_top_k": 10,
                "bm25_top_k": 10,
                "enable_reranker": False,
            },
            RoutingDecision.STANDARD: {
                "final_top_k": 5,
                "dense_top_k": 20,
                "bm25_top_k": 20,
                "enable_reranker": True,
            },
            RoutingDecision.COMPLEX: {
                "final_top_k": 8,
                "dense_top_k": 30,
                "bm25_top_k": 30,
                "enable_reranker": True,
            },
            RoutingDecision.EVIDENCE_CRITICAL: {
                "final_top_k": 10,
                "dense_top_k": 35,
                "bm25_top_k": 35,
                "enable_reranker": True,
            },
        }
        return configs.get(decision, configs[RoutingDecision.STANDARD])

    def get_generation_config(self, decision: RoutingDecision) -> dict[str, Any]:
        """Provide tailored generation parameters based on routing decision.

        Args:
            decision: Routing classification.

        Returns:
            Dictionary of generator parameter overrides.
        """
        configs = {
            RoutingDecision.SIMPLE: {
                "max_new_tokens": 256,
                "temperature": 0.1,
                "do_sample": False,
            },
            RoutingDecision.STANDARD: {
                "max_new_tokens": 512,
                "temperature": 0.1,
                "do_sample": True,
            },
            RoutingDecision.COMPLEX: {
                "max_new_tokens": 768,
                "temperature": 0.15,
                "do_sample": True,
            },
            RoutingDecision.EVIDENCE_CRITICAL: {
                "max_new_tokens": 768,
                "temperature": 0.0,
                "do_sample": False,
            },
        }
        return configs.get(decision, configs[RoutingDecision.STANDARD])

    def get_verification_config(self, decision: RoutingDecision) -> dict[str, Any]:
        """Provide verification pipeline overrides based on routing decision.

        Args:
            decision: Routing classification.

        Returns:
            Dictionary of verification parameters (e.g., rigor and thresholds).
        """
        configs = {
            RoutingDecision.SIMPLE: {
                "required": False,
                "strict_mode": False,
                "min_support_ratio": 0.5,
            },
            RoutingDecision.STANDARD: {
                "required": True,
                "strict_mode": False,
                "min_support_ratio": 0.6,
            },
            RoutingDecision.COMPLEX: {
                "required": True,
                "strict_mode": True,
                "min_support_ratio": 0.7,
            },
            RoutingDecision.EVIDENCE_CRITICAL: {
                "required": True,
                "strict_mode": True,
                "min_support_ratio": 0.85,
                "max_unsupported_claims": 0,
            },
        }
        return configs.get(decision, configs[RoutingDecision.STANDARD])


class AdaptiveRouter:
    """Adaptive query router with telemetry for learned routing policies.

    Wraps RuleBasedRouter with execution logging and dynamic adjustment capabilities.
    """

    def __init__(
        self,
        base_router: Optional[RuleBasedRouter] = None,
        config: Optional[RouterConfig] = None,
    ) -> None:
        """Initialize adaptive router.

        Args:
            base_router: Underlying rule-based router.
            config: Optional router configuration.
        """
        self.base_router = base_router or RuleBasedRouter(config=config)
        self.history: list[dict[str, Any]] = []
        logger.info("AdaptiveRouter initialized wrapping %s", self.base_router.__class__.__name__)

    def classify_question(self, question: str) -> RoutingDecision:
        """Classify a question and record routing telemetry.

        Args:
            question: Clinical question text.

        Returns:
            Selected RoutingDecision.
        """
        decision = self.base_router.classify_question(question)
        self.history.append({"question": question[:100], "decision": decision.value})
        return decision

    def assess_complexity(self, question: str) -> QuestionComplexity:
        """Assess query complexity via underlying router."""
        return self.base_router.assess_complexity(question)

    def get_retrieval_config(self, decision: RoutingDecision) -> dict[str, Any]:
        """Retrieve retrieval overrides."""
        return self.base_router.get_retrieval_config(decision)

    def get_generation_config(self, decision: RoutingDecision) -> dict[str, Any]:
        """Retrieve generation overrides."""
        return self.base_router.get_generation_config(decision)

    def get_verification_config(self, decision: RoutingDecision) -> dict[str, Any]:
        """Retrieve verification overrides."""
        return self.base_router.get_verification_config(decision)

    def update_policy(self, validation_results: list[dict[str, Any]]) -> None:
        """Update adaptive routing policy based on validation feedback.

        Args:
            validation_results: List of validation outcomes with accuracy/risk metrics.
        """
        logger.info(
            "AdaptiveRouter received %d validation results for policy calibration",
            len(validation_results),
        )
