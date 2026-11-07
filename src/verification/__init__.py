"""Verification package for MedFusion-RAFT."""
from src.verification.claim_extractor import ClaimExtractor, ClaimExtractionConfig
from src.verification.citation_validator import (
    CitationValidator,
    CitationValidationConfig,
    CitationValidationResult,
    OverallCitationResult,
)
from src.verification.entailment import (
    EntailmentChecker,
    EntailmentConfig,
    EntailmentResult,
    MockEntailmentChecker,
)
from src.verification.pipeline import VerificationPipeline, VerificationPipelineConfig
from src.verification.decision import (
    DecisionEngine,
    DecisionPolicyConfig,
    AbstentionConfig,
    CalibrationModel,
)

__all__ = [
    "ClaimExtractor",
    "ClaimExtractionConfig",
    "CitationValidator",
    "CitationValidationConfig",
    "CitationValidationResult",
    "OverallCitationResult",
    "EntailmentChecker",
    "EntailmentConfig",
    "EntailmentResult",
    "MockEntailmentChecker",
    "VerificationPipeline",
    "VerificationPipelineConfig",
    "DecisionEngine",
    "DecisionPolicyConfig",
    "AbstentionConfig",
    "CalibrationModel",
]
