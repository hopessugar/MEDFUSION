"""Configuration loader and schema definitions for MedFusion-RAFT.

Provides type-safe configuration parsing using Pydantic, supports YAML loading,
environment variable overrides, singleton caching, and experiment overrides.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

from src.models.schemas import ExperimentConfig


class QuantizationConfig(BaseModel):
    """BitsAndBytes 4-bit/8-bit quantization configuration."""

    enabled: bool = True
    bits: int = 4
    quant_type: str = "nf4"
    compute_dtype: str = "bfloat16"
    use_double_quant: bool = True


class GenerationConfig(BaseModel):
    """Text generation parameters for causal language models."""

    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.0


class GeneratorConfig(BaseModel):
    """Configuration for the medical generator backbone (MedGemma)."""

    model_id: str = "google/medgemma-1.5-4b-it"
    model_type: str = "medgemma"
    revision: str = "main"
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    device_map: str = "auto"
    trust_remote_code: bool = True


class VerifierConfig(BaseModel):
    """Configuration for the verifier backbone (Qwen3.5)."""

    model_id: str = "Qwen/Qwen3.5-9B"
    model_type: str = "qwen"
    revision: str = "main"
    load_as_causal_lm: bool = True
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    generation: GenerationConfig = Field(default_factory=lambda: GenerationConfig(max_new_tokens=1024))
    device_map: str = "auto"
    trust_remote_code: bool = True


class RetrievalEncoderConfig(BaseModel):
    """Models and dimensions for MedCPT retrieval components."""

    query_encoder: str = "ncbi/MedCPT-Query-Encoder"
    article_encoder: str = "ncbi/MedCPT-Article-Encoder"
    cross_encoder: str = "ncbi/MedCPT-Cross-Encoder"
    embedding_dim: int = 768
    max_query_length: int = 64
    max_article_length: int = 512


class ModelsConfig(BaseModel):
    """Aggregated models configuration (models.yaml)."""

    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)
    verifier: VerifierConfig = Field(default_factory=VerifierConfig)
    retrieval: RetrievalEncoderConfig = Field(default_factory=RetrievalEncoderConfig)


class QLoRAConfig(BaseModel):
    """PEFT QLoRA adapter hyperparameters."""

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: Optional[List[str]] = None


class SFTTrainingConfig(BaseModel):
    """Supervised fine-tuning parameters."""

    learning_rate: float = 2.0e-5
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_seq_length: int = 2048
    optimizer: str = "paged_adamw_8bit"
    lr_scheduler_type: str = "cosine"
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 50
    save_total_limit: int = 3
    fp16: bool = False
    bf16: bool = True
    seed: int = 42
    output_dir: str = "./experiments/checkpoints/sft"


class RAFTTrainingConfig(SFTTrainingConfig):
    """Retrieval-Augmented Fine-Tuning specific hyperparameters."""

    learning_rate: float = 1.0e-5
    distractor_ratio: float = 0.5
    num_distractor_passages: int = 3
    output_dir: str = "./experiments/checkpoints/raft"


class TrainingConfig(BaseModel):
    """Aggregated training configuration (training.yaml)."""

    qlora: QLoRAConfig = Field(default_factory=QLoRAConfig)
    sft: SFTTrainingConfig = Field(default_factory=SFTTrainingConfig)
    raft: RAFTTrainingConfig = Field(default_factory=RAFTTrainingConfig)


class ChunkingConfig(BaseModel):
    """Text chunking parameters for document ingestion."""

    method: str = "structure_aware"
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 100
    preserve_sections: bool = True
    preserve_sentences: bool = True


class BM25Config(BaseModel):
    """BM25 sparse retrieval parameters."""

    k1: float = 1.5
    b: float = 0.75
    top_k: int = 50


class DenseRetrievalConfig(BaseModel):
    """Dense FAISS retrieval parameters."""

    top_k: int = 50
    batch_size: int = 32
    index_type: str = "flat"
    nprobe: int = 10


class FusionConfig(BaseModel):
    """Rank fusion (e.g. Reciprocal Rank Fusion) parameters."""

    method: str = "rrf"
    rrf_k: int = 60
    top_k: int = 20


class RerankingConfig(BaseModel):
    """Cross-encoder reranking parameters."""

    enabled: bool = True
    top_k: int = 5
    batch_size: int = 16


class RetrievalConfig(BaseModel):
    """Aggregated retrieval pipeline configuration (retrieval.yaml)."""

    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    bm25: BM25Config = Field(default_factory=BM25Config)
    dense: DenseRetrievalConfig = Field(default_factory=DenseRetrievalConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)


class ClaimExtractionConfig(BaseModel):
    """Decomposition configuration for atomic claim extraction."""

    max_claims_per_answer: int = 20
    min_claim_length: int = 10


class CitationValidationConfig(BaseModel):
    """Rules for checking citations against source passages."""

    strict_mode: bool = True
    check_passage_existence: bool = True
    check_content_relevance: bool = True


class EntailmentConfig(BaseModel):
    """NLI entailment classifier thresholds."""

    model: str = "cross-encoder/nli-deberta-v3-base"
    threshold_supported: float = 0.7
    threshold_contradicted: float = 0.3


class VerificationPolicyConfig(BaseModel):
    """Support ratio criteria for factual verification."""

    require_all_claims_supported: bool = False
    min_support_ratio: float = 0.6
    max_unsupported_claims: int = 2
    max_contradicted_claims: int = 0


class AbstentionConfig(BaseModel):
    """Thresholds for selective prediction and abstention."""

    enabled: bool = True
    confidence_threshold: float = 0.5
    min_evidence_passages: int = 1
    max_revision_attempts: int = 1
    calibration_method: str = "platt_scaling"


class DecisionPolicyConfig(BaseModel):
    """Decision boundaries for accepting, revising, or abstaining."""

    accept_min_score: float = 0.7
    revise_min_score: float = 0.4
    abstain_below: float = 0.4


class VerificationSubsystemConfig(BaseModel):
    """Aggregated verification configuration (verification.yaml)."""

    claim_extraction: ClaimExtractionConfig = Field(default_factory=ClaimExtractionConfig)
    citation_validation: CitationValidationConfig = Field(default_factory=CitationValidationConfig)
    entailment: EntailmentConfig = Field(default_factory=EntailmentConfig)
    verification: VerificationPolicyConfig = Field(default_factory=VerificationPolicyConfig)
    abstention: AbstentionConfig = Field(default_factory=AbstentionConfig)
    decision_policy: DecisionPolicyConfig = Field(default_factory=DecisionPolicyConfig)


class AppEnvironmentConfig(BaseModel):
    """System-level environment paths and variables."""

    hf_token: Optional[str] = None
    mlflow_tracking_uri: str = "./experiments/mlruns"
    cuda_visible_devices: str = "0"
    log_level: str = "INFO"
    config_dir: str = "./configs"
    data_dir: str = "./data"
    experiment_dir: str = "./experiments"
    model_cache_dir: str = "./models"


class AppConfig(BaseModel):
    """Complete root configuration object for MedFusion-RAFT."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    env: AppEnvironmentConfig = Field(default_factory=AppEnvironmentConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    verification: VerificationSubsystemConfig = Field(
        default_factory=VerificationSubsystemConfig
    )
    evaluation: Dict[str, Any] = Field(default_factory=dict)
    experiments: Dict[str, ExperimentConfig] = Field(default_factory=dict)

    def get_experiment(self, experiment_id: str) -> ExperimentConfig:
        """Fetch an experiment specification by ID.

        Args:
            experiment_id: Experiment key (e.g., 'E1_base', 'E5_raft').

        Returns:
            ExperimentConfig instance.

        Raises:
            KeyError: If experiment_id is not defined.
        """
        if experiment_id not in self.experiments:
            available = list(self.experiments.keys())
            raise KeyError(
                f"Experiment '{experiment_id}' not found. Available: {available}"
            )
        return self.experiments[experiment_id]


def _read_yaml_file(path: Path) -> Dict[str, Any]:
    """Read a YAML file safely, returning an empty dict if not found."""
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}


def load_app_config(config_dir: Optional[Path | str] = None) -> AppConfig:
    """Load and validate all configuration files from disk with env overrides.

    Args:
        config_dir: Directory containing YAML files. If None, resolves from
                    MEDFUSION_CONFIG_DIR env var or defaults to './configs'.

    Returns:
        Fully initialized and validated AppConfig.
    """
    if config_dir is None:
        raw_dir = os.environ.get("MEDFUSION_CONFIG_DIR", "./configs")
        config_path = Path(raw_dir).resolve()
    else:
        config_path = Path(config_dir).resolve()

    # Build environment config
    env_cfg = AppEnvironmentConfig(
        hf_token=os.environ.get("HF_TOKEN"),
        mlflow_tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "./experiments/mlruns"),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        log_level=os.environ.get("MEDFUSION_LOG_LEVEL", "INFO"),
        config_dir=str(config_path),
        data_dir=os.environ.get("MEDFUSION_DATA_DIR", "./data"),
        experiment_dir=os.environ.get("MEDFUSION_EXPERIMENT_DIR", "./experiments"),
        model_cache_dir=os.environ.get("MEDFUSION_MODEL_CACHE_DIR", "./models"),
    )

    models_data = _read_yaml_file(config_path / "models.yaml")
    training_data = _read_yaml_file(config_path / "training.yaml")
    retrieval_data = _read_yaml_file(config_path / "retrieval.yaml")
    verification_data = _read_yaml_file(config_path / "verification.yaml")
    evaluation_data = _read_yaml_file(config_path / "evaluation.yaml")
    experiments_raw = _read_yaml_file(config_path / "experiments.yaml")

    # Parse experiment configurations
    parsed_experiments: Dict[str, ExperimentConfig] = {}
    exp_dict = experiments_raw.get("experiments", {})
    for exp_id, exp_spec in exp_dict.items():
        if isinstance(exp_spec, dict):
            parsed_experiments[exp_id] = ExperimentConfig(
                experiment_id=exp_id,
                name=exp_spec.get("name", exp_id),
                generator=exp_spec.get("generator", "base"),
                retrieval=exp_spec.get("retrieval"),
                fine_tuning=exp_spec.get("fine_tuning"),
                verification=exp_spec.get("verification", False),
                abstention=exp_spec.get("abstention", False),
                routing=exp_spec.get("routing", False),
                seed=exp_spec.get("seed", 42),
                notes=exp_spec.get("notes"),
            )

    return AppConfig(
        env=env_cfg,
        models=ModelsConfig(**models_data) if models_data else ModelsConfig(),
        training=TrainingConfig(**training_data) if training_data else TrainingConfig(),
        retrieval=RetrievalConfig(**retrieval_data) if retrieval_data else RetrievalConfig(),
        verification=(
            VerificationSubsystemConfig(**verification_data)
            if verification_data
            else VerificationSubsystemConfig()
        ),
        evaluation=evaluation_data,
        experiments=parsed_experiments,
    )


_GLOBAL_CONFIG: Optional[AppConfig] = None


def get_app_config(
    reload: bool = False, config_dir: Optional[Path | str] = None
) -> AppConfig:
    """Retrieve singleton AppConfig instance.

    Args:
        reload: Force re-reading from disk if True.
        config_dir: Optional custom configuration directory.

    Returns:
        Singleton AppConfig.
    """
    global _GLOBAL_CONFIG
    if _GLOBAL_CONFIG is None or reload:
        _GLOBAL_CONFIG = load_app_config(config_dir=config_dir)
    return _GLOBAL_CONFIG
