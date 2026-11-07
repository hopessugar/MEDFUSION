"""Abstract base classes and core data models for model providers in MedFusion-RAFT."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------

class ModelError(Exception):
    """Base exception for all model provider errors."""


class ModelLoadError(ModelError):
    """Raised when a model or tokenizer fails to load."""


class ModelInferenceError(ModelError):
    """Raised when model inference or generation fails."""


class OutputParsingError(ModelError):
    """Raised when structured model output cannot be parsed."""


# ---------------------------------------------------------------------------
# Enumerations for Structured Medical Outputs
# ---------------------------------------------------------------------------

class AnswerabilityStatus(str, Enum):
    """Answerability classification for medical queries."""

    ANSWERABLE = "answerable"
    PARTIALLY_ANSWERABLE = "partially_answerable"
    UNANSWERABLE = "unanswerable"


class ClaimVerificationStatus(str, Enum):
    """Verification classification for individual factual claims."""

    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    CONTRADICTED = "CONTRADICTED"


class OverallAssessmentStatus(str, Enum):
    """Overall verification assessment for an entire answer."""

    VERIFIED = "VERIFIED"
    PARTIALLY_VERIFIED = "PARTIALLY_VERIFIED"
    NOT_VERIFIED = "NOT_VERIFIED"


# ---------------------------------------------------------------------------
# Configuration Models
# ---------------------------------------------------------------------------

class ModelConfig(BaseModel):
    """Configuration for a model provider.

    Attributes:
        model_id: Hugging Face model identifier or local directory path.
        revision: Model repository revision (e.g., 'main', branch name, commit sha).
        device_map: Accelerate device mapping strategy (e.g., 'auto', 'cpu', 'cuda:0').
        trust_remote_code: Whether to allow remote code execution from HF models.
        quantization_bits: Optional bit width for quantization (4, 8, or None).
        quant_type: Quantization type for 4-bit (e.g., 'nf4', 'fp4').
        compute_dtype: Target computation dtype ('bfloat16', 'float16', 'float32').
        use_double_quant: Whether to use nested quantization in 4-bit mode.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (higher = more creative, lower = more deterministic).
        top_p: Nucleus sampling probability cutoff.
        do_sample: Whether to use sampling during generation.
        repetition_penalty: Penalty applied for token repetition.
        adapter_path: Optional path to LoRA/PEFT adapter weights.
        provider_type: Optional provider selector tag ('medgemma', 'qwen', 'mock').
        extra_kwargs: Additional provider-specific keyword arguments.
    """

    model_config = ConfigDict(extra="ignore")

    model_id: str
    revision: str = "main"
    device_map: str = "auto"
    trust_remote_code: bool = True
    quantization_bits: Optional[int] = None
    quant_type: str = "nf4"
    compute_dtype: str = "bfloat16"
    use_double_quant: bool = True
    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.1
    adapter_path: Optional[str] = None
    provider_type: Optional[str] = None
    extra_kwargs: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> ModelConfig:
        """Load configuration from a YAML file.

        Args:
            path: Absolute or relative path to the YAML file.

        Returns:
            Instantiated ModelConfig.

        Raises:
            FileNotFoundError: If the config file does not exist.
            ValueError: If the YAML content cannot be parsed.
        """
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found at: {config_path}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return cls(**data)
        except Exception as exc:
            logger.error("Failed to parse YAML config from %s: %s", config_path, exc)
            raise ValueError(f"Invalid YAML config at {config_path}: {exc}") from exc

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to a YAML file.

        Args:
            path: Destination file path.
        """
        dest_path = Path(path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Generation Outputs & Parsed Models
# ---------------------------------------------------------------------------

class GenerationOutput(BaseModel):
    """Output from text generation.

    Attributes:
        text: Raw generated text.
        input_tokens: Number of prompt/input tokens.
        output_tokens: Number of generated completion tokens.
        generation_time_ms: Total inference latency in milliseconds.
        metadata: Arbitrary metadata dictionary for logging and profiling.
    """

    model_config = ConfigDict(extra="ignore")

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    generation_time_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedMedicalAnswer(BaseModel):
    """Structured representation of a parsed medical answer."""

    model_config = ConfigDict(extra="ignore")

    answer: str
    citations: list[str] = Field(default_factory=list)
    answerability: AnswerabilityStatus = AnswerabilityStatus.ANSWERABLE
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    raw_text: str = ""


class VerificationClaim(BaseModel):
    """Individual atomic claim verification assessment."""

    model_config = ConfigDict(extra="ignore")

    claim: str
    status: ClaimVerificationStatus = ClaimVerificationStatus.SUPPORTED
    evidence: str = "NONE"
    reasoning: str = ""


class ParsedVerificationResult(BaseModel):
    """Structured representation of answer verification results."""

    model_config = ConfigDict(extra="ignore")

    claims: list[VerificationClaim] = Field(default_factory=list)
    overall_assessment: OverallAssessmentStatus = OverallAssessmentStatus.VERIFIED
    support_score: float = Field(default=1.0, ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)
    raw_text: str = ""


# ---------------------------------------------------------------------------
# Abstract Base Providers
# ---------------------------------------------------------------------------

class BaseModelProvider(ABC):
    """Abstract base class for all model providers."""

    def __init__(self, config: ModelConfig) -> None:
        """Initialize provider with configuration.

        Args:
            config: Provider configuration instance.
        """
        self.config = config
        self._model: Any = None
        self._tokenizer: Any = None
        self._loaded: bool = False

    @abstractmethod
    def load(self) -> None:
        """Load the model and tokenizer into memory."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Unload the model from memory and free GPU/system resources."""
        ...

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> GenerationOutput:
        """Generate text from a prompt.

        Args:
            prompt: Text prompt for generation.
            **kwargs: Generation override parameters.

        Returns:
            GenerationOutput containing text and metadata.
        """
        ...

    @property
    def is_loaded(self) -> bool:
        """Check whether the model is currently loaded in memory."""
        return self._loaded

    @abstractmethod
    def get_memory_usage(self) -> dict[str, float]:
        """Return current memory usage statistics in MB.

        Returns:
            Dictionary with memory keys (e.g., 'allocated_mb', 'reserved_mb').
        """
        ...

    def __enter__(self) -> BaseModelProvider:
        """Context manager entry to load the model."""
        if not self.is_loaded:
            self.load()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit to unload the model."""
        self.unload()


class BaseGeneratorProvider(BaseModelProvider):
    """Base class for answer generation models."""

    @abstractmethod
    def generate_answer(self, question: str, context: str, **kwargs: Any) -> GenerationOutput:
        """Generate a medical answer given question and context.

        Args:
            question: Clinical or medical question.
            context: Retrieved evidence context string.
            **kwargs: Additional generation options.

        Returns:
            GenerationOutput containing answer text and metrics.
        """
        ...


class BaseVerifierProvider(BaseModelProvider):
    """Base class for verification models."""

    @abstractmethod
    def verify_answer(
        self,
        question: str,
        answer: str,
        evidence: list[str],
        **kwargs: Any,
    ) -> GenerationOutput:
        """Verify an answer against evidence.

        Args:
            question: Original question.
            answer: Answer generated by candidate model.
            evidence: List of evidence passage strings.
            **kwargs: Additional generation options.

        Returns:
            GenerationOutput containing verification text and metrics.
        """
        ...
