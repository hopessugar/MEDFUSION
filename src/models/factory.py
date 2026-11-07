"""Factory functions and registries for creating MedFusion-RAFT model providers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type, Union

from src.models.base import (
    BaseGeneratorProvider,
    BaseModelProvider,
    BaseVerifierProvider,
    ModelConfig,
    ModelError,
)
from src.models.medgemma import MedGemmaProvider
from src.models.mock_providers import MockGeneratorProvider, MockVerifierProvider
from src.models.qwen_verifier import QwenVerifierProvider

logger = logging.getLogger(__name__)

# Registry types
GeneratorFactoryType = Type[BaseGeneratorProvider]
VerifierFactoryType = Type[BaseVerifierProvider]

# Default provider registries
_GENERATOR_REGISTRY: Dict[str, GeneratorFactoryType] = {
    "medgemma": MedGemmaProvider,
    "mock": MockGeneratorProvider,
    "mock_generator": MockGeneratorProvider,
}

_VERIFIER_REGISTRY: Dict[str, VerifierFactoryType] = {
    "qwen": QwenVerifierProvider,
    "mock": MockVerifierProvider,
    "mock_verifier": MockVerifierProvider,
}


def register_generator_provider(name: str, provider_cls: GeneratorFactoryType) -> None:
    """Register a new generator provider class.

    Args:
        name: Unique provider lookup key (case-insensitive).
        provider_cls: Provider class subclassing BaseGeneratorProvider.
    """
    key = name.strip().lower()
    _GENERATOR_REGISTRY[key] = provider_cls
    logger.info("Registered generator provider: '%s' -> %s", key, provider_cls.__name__)


def register_verifier_provider(name: str, provider_cls: VerifierFactoryType) -> None:
    """Register a new verifier provider class.

    Args:
        name: Unique provider lookup key (case-insensitive).
        provider_cls: Provider class subclassing BaseVerifierProvider.
    """
    key = name.strip().lower()
    _VERIFIER_REGISTRY[key] = provider_cls
    logger.info("Registered verifier provider: '%s' -> %s", key, provider_cls.__name__)


def _resolve_model_config(
    config: Union[ModelConfig, dict[str, Any], str, Path, None],
    default_model_id: str,
    default_provider: str,
    **kwargs: Any,
) -> ModelConfig:
    """Normalize various configuration inputs into a validated ModelConfig instance.

    Args:
        config: ModelConfig instance, YAML path, dictionary, or model identifier string.
        default_model_id: Fallback model identifier.
        default_provider: Fallback provider tag.
        **kwargs: Overrides to pass into ModelConfig.

    Returns:
        Validated ModelConfig instance.
    """
    if config is None:
        return ModelConfig(
            model_id=default_model_id,
            provider_type=default_provider,
            **kwargs,
        )

    if isinstance(config, ModelConfig):
        # Apply any overrides from kwargs if provided
        if kwargs:
            data = config.model_dump()
            data.update(kwargs)
            return ModelConfig(**data)
        return config

    if isinstance(config, (str, Path)):
        candidate_path = Path(config)
        if candidate_path.is_file():
            logger.info("Loading model config from YAML: %s", candidate_path)
            loaded = ModelConfig.from_yaml(candidate_path)
            if kwargs:
                data = loaded.model_dump()
                data.update(kwargs)
                return ModelConfig(**data)
            return loaded
        else:
            # Treat as model_id string
            return ModelConfig(
                model_id=str(config),
                provider_type=kwargs.pop("provider_type", default_provider),
                **kwargs,
            )

    if isinstance(config, dict):
        merged = {**config, **kwargs}
        return ModelConfig(**merged)

    raise ValueError(f"Unsupported config type: {type(config).__name__}")


def _detect_provider_type(
    config: ModelConfig,
    explicit_type: Optional[str] = None,
    available_keys: Optional[set[str]] = None,
) -> str:
    """Infer provider key from explicit override, config field, or model_id naming.

    Args:
        config: ModelConfig instance.
        explicit_type: Optional explicit provider key.
        available_keys: Set of valid provider keys.

    Returns:
        Normalized provider key string.
    """
    if explicit_type:
        return explicit_type.strip().lower()

    if config.provider_type:
        return config.provider_type.strip().lower()

    mid = config.model_id.lower()
    if "mock" in mid:
        return "mock"
    if "gemma" in mid or "medgemma" in mid:
        return "medgemma"
    if "qwen" in mid:
        return "qwen"

    # If key matching available keys directly
    if available_keys and mid in available_keys:
        return mid

    return "mock"


def create_generator(
    config: Union[ModelConfig, dict[str, Any], str, Path, None] = None,
    provider_type: Optional[str] = None,
    **kwargs: Any,
) -> BaseGeneratorProvider:
    """Instantiate a generator provider based on configuration.

    Args:
        config: ModelConfig, YAML path, dictionary, model ID string, or None.
        provider_type: Optional explicit provider identifier ('medgemma', 'mock').
        **kwargs: Extra parameters passed to configuration or provider constructor.

    Returns:
        Configured BaseGeneratorProvider instance.

    Raises:
        ModelError: If provider type is unregistered or invalid.
    """
    resolved_config = _resolve_model_config(
        config=config,
        default_model_id="google/medgemma-4b-it",
        default_provider="medgemma",
        **kwargs,
    )

    selected_type = _detect_provider_type(
        resolved_config,
        explicit_type=provider_type,
        available_keys=set(_GENERATOR_REGISTRY.keys()),
    )

    provider_cls = _GENERATOR_REGISTRY.get(selected_type)
    if provider_cls is None:
        valid_options = list(_GENERATOR_REGISTRY.keys())
        raise ModelError(
            f"Unknown generator provider '{selected_type}'. Available providers: {valid_options}"
        )

    logger.info(
        "Creating generator provider: type='%s', class=%s, model_id='%s'",
        selected_type,
        provider_cls.__name__,
        resolved_config.model_id,
    )

    return provider_cls(config=resolved_config)


def create_verifier(
    config: Union[ModelConfig, dict[str, Any], str, Path, None] = None,
    provider_type: Optional[str] = None,
    **kwargs: Any,
) -> BaseVerifierProvider:
    """Instantiate a verifier provider based on configuration.

    Args:
        config: ModelConfig, YAML path, dictionary, model ID string, or None.
        provider_type: Optional explicit provider identifier ('qwen', 'mock').
        **kwargs: Extra parameters passed to configuration or provider constructor.

    Returns:
        Configured BaseVerifierProvider instance.

    Raises:
        ModelError: If provider type is unregistered or invalid.
    """
    resolved_config = _resolve_model_config(
        config=config,
        default_model_id="Qwen/Qwen2.5-7B-Instruct",
        default_provider="qwen",
        **kwargs,
    )

    selected_type = _detect_provider_type(
        resolved_config,
        explicit_type=provider_type,
        available_keys=set(_VERIFIER_REGISTRY.keys()),
    )

    provider_cls = _VERIFIER_REGISTRY.get(selected_type)
    if provider_cls is None:
        valid_options = list(_VERIFIER_REGISTRY.keys())
        raise ModelError(
            f"Unknown verifier provider '{selected_type}'. Available providers: {valid_options}"
        )

    logger.info(
        "Creating verifier provider: type='%s', class=%s, model_id='%s'",
        selected_type,
        provider_cls.__name__,
        resolved_config.model_id,
    )

    return provider_cls(config=resolved_config)
