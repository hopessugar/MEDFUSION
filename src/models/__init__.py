"""Model provider layer for MedFusion-RAFT.

Exports abstract base classes, concrete providers (MedGemma, Qwen verifier, mock),
structured data models, and factory functions for model instantiation.
"""

from __future__ import annotations

from src.models.base import (
    AnswerabilityStatus,
    BaseGeneratorProvider,
    BaseModelProvider,
    BaseVerifierProvider,
    ClaimVerificationStatus,
    GenerationOutput,
    ModelConfig,
    ModelError,
    ModelInferenceError,
    ModelLoadError,
    OutputParsingError,
    OverallAssessmentStatus,
    ParsedMedicalAnswer,
    ParsedVerificationResult,
    VerificationClaim,
)
from src.models.factory import (
    create_generator,
    create_verifier,
    register_generator_provider,
    register_verifier_provider,
)
from src.models.medgemma import MedGemmaProvider
from src.models.mock_providers import MockGeneratorProvider, MockVerifierProvider
from src.models.qwen_verifier import QwenVerifierProvider

__all__ = [
    # Base and Configuration
    "ModelConfig",
    "GenerationOutput",
    "BaseModelProvider",
    "BaseGeneratorProvider",
    "BaseVerifierProvider",
    # Enums and Structured Outputs
    "AnswerabilityStatus",
    "ClaimVerificationStatus",
    "OverallAssessmentStatus",
    "ParsedMedicalAnswer",
    "VerificationClaim",
    "ParsedVerificationResult",
    # Exceptions
    "ModelError",
    "ModelLoadError",
    "ModelInferenceError",
    "OutputParsingError",
    # Providers
    "MedGemmaProvider",
    "QwenVerifierProvider",
    "MockGeneratorProvider",
    "MockVerifierProvider",
    # Factories
    "create_generator",
    "create_verifier",
    "register_generator_provider",
    "register_verifier_provider",
]
