"""Retrieval-Augmented Fine-Tuning (RAFT) trainer for MedFusion-RAFT.

Extends the SFT training approach to RAFT-formatted data, supporting
prior SFT adapter loading, RAFT-specific data mix validation, and
lower default learning rates suited to the RAFT fine-tuning stage.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.training.qlora.config import (
    QLoRAConfig,
    get_bnb_config,
    prepare_model_for_qlora,
)
from src.training.utils import (
    check_gpu_memory,
    create_training_args_from_config,
    get_hardware_info,
    log_training_config,
    set_seed,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Optional dependency imports
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]
    TRANSFORMERS_AVAILABLE = False

try:
    from peft import PeftModel

    PEFT_AVAILABLE = True
except ImportError:
    PeftModel = None  # type: ignore[assignment]
    PEFT_AVAILABLE = False

try:
    from trl import SFTTrainer, SFTConfig

    TRL_AVAILABLE = True
except ImportError:
    SFTTrainer = None  # type: ignore[assignment]
    SFTConfig = None  # type: ignore[assignment]
    TRL_AVAILABLE = False

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None  # type: ignore[assignment]
    MLFLOW_AVAILABLE = False


class RAFTTrainerConfig(BaseModel):
    """Configuration for the MedFusion RAFT trainer.

    Inherits most fields from SFT training with RAFT-specific defaults:
    lower learning rate, distractor ratio tracking, and optional SFT adapter path.

    Attributes:
        model_id: HuggingFace model identifier or local path.
        sft_adapter_path: Optional path to a pre-trained SFT LoRA adapter to
            load before RAFT fine-tuning. Enables SFT→RAFT two-stage training.
        output_dir: Directory for checkpoints and outputs.
        learning_rate: Peak learning rate (lower default for RAFT stage).
        num_train_epochs: Number of training epochs.
        per_device_train_batch_size: Training batch size per GPU.
        gradient_accumulation_steps: Steps before optimizer update.
        warmup_ratio: Fraction of steps for LR warmup.
        weight_decay: AdamW weight decay coefficient.
        max_seq_length: Maximum sequence length.
        optimizer: Optimizer name.
        lr_scheduler_type: Learning rate scheduler type.
        gradient_checkpointing: Enable gradient checkpointing.
        logging_steps: Log metrics every N steps.
        save_steps: Save checkpoint every N steps.
        eval_steps: Evaluate every N steps.
        save_total_limit: Maximum saved checkpoints.
        fp16: Use FP16 mixed precision.
        bf16: Use BF16 mixed precision.
        seed: Random seed.
        quantization_bits: Quantization bit-width (0=disabled).
        trust_remote_code: Trust remote code from HF hub.
        mlflow_experiment: MLflow experiment name.
        device_map: Device placement strategy.
        distractor_ratio: Expected ratio of distractor examples for validation.
        min_answerable_ratio: Minimum required ratio of answerable examples.
    """

    model_id: str = Field(
        default="google/medgemma-4b-it",
        description="HuggingFace model ID or local path",
    )
    sft_adapter_path: Optional[str] = Field(
        default=None,
        description="Path to pre-trained SFT adapter for two-stage training",
    )
    output_dir: str = Field(
        default="./experiments/checkpoints/raft",
        description="Checkpoint output directory",
    )
    learning_rate: float = Field(default=1e-5, description="Peak learning rate (RAFT default)")
    num_train_epochs: int = Field(default=2, ge=1, description="Training epochs")
    per_device_train_batch_size: int = Field(default=1, ge=1, description="Batch size per device")
    gradient_accumulation_steps: int = Field(default=8, ge=1, description="Gradient accumulation")
    warmup_ratio: float = Field(default=0.1, ge=0.0, le=1.0, description="LR warmup ratio")
    weight_decay: float = Field(default=0.01, ge=0.0, description="Weight decay")
    max_seq_length: int = Field(default=2048, ge=64, description="Max sequence length")
    optimizer: str = Field(default="paged_adamw_8bit", description="Optimizer name")
    lr_scheduler_type: str = Field(default="cosine", description="LR scheduler")
    gradient_checkpointing: bool = Field(default=True, description="Enable gradient checkpointing")
    logging_steps: int = Field(default=10, ge=1, description="Log every N steps")
    save_steps: int = Field(default=100, ge=1, description="Save every N steps")
    eval_steps: int = Field(default=50, ge=1, description="Eval every N steps")
    save_total_limit: int = Field(default=3, ge=1, description="Max checkpoints")
    fp16: bool = Field(default=False, description="Use FP16")
    bf16: bool = Field(default=True, description="Use BF16")
    seed: int = Field(default=42, description="Random seed")
    quantization_bits: int = Field(default=4, description="Quantization bits")
    trust_remote_code: bool = Field(default=True, description="Trust remote code")
    mlflow_experiment: Optional[str] = Field(
        default=None, description="MLflow experiment name"
    )
    device_map: str = Field(default="auto", description="Device map strategy")
    distractor_ratio: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Expected distractor-only example ratio",
    )
    min_answerable_ratio: float = Field(
        default=0.3, ge=0.0, le=1.0,
        description="Minimum required answerable example ratio",
    )


class MedFusionRAFTTrainer:
    """RAFT trainer for retrieval-augmented medical QA fine-tuning.

    Supports two-stage training (SFT→RAFT) by optionally loading a previously
    trained SFT adapter, then applying RAFT-specific training on data that
    includes gold and distractor passages.
    """

    def __init__(
        self,
        trainer_config: RAFTTrainerConfig,
        qlora_config: Optional[QLoRAConfig] = None,
    ) -> None:
        """Initialize the RAFT trainer.

        Args:
            trainer_config: RAFT training configuration.
            qlora_config: QLoRA adapter configuration. Uses defaults if None.
        """
        self.trainer_config = trainer_config
        self.qlora_config = qlora_config or QLoRAConfig()
        self.model: Any = None
        self.tokenizer: Any = None
        self.trainer: Any = None
        self._is_prepared = False

    def _validate_training_data(self, dataset: Any) -> dict[str, Any]:
        """Validate that the training data has a proper RAFT mix.

        Checks category distribution, ensures sufficient answerable examples,
        and verifies all records have non-empty text.

        Args:
            dataset: Training dataset (HF Dataset or list of dicts).

        Returns:
            Validation report dictionary.

        Raises:
            ValueError: If data mix fails minimum quality requirements.
        """
        items = list(dataset) if not isinstance(dataset, list) else dataset
        total = len(items)

        if total == 0:
            raise ValueError("Training dataset is empty")

        category_counts: dict[str, int] = {}
        empty_texts = 0

        for item in items:
            cat = item.get("category", "unknown") if isinstance(item, dict) else "unknown"
            category_counts[cat] = category_counts.get(cat, 0) + 1

            text = item.get("text", "") if isinstance(item, dict) else str(item)
            if not text or not str(text).strip():
                empty_texts += 1

        answerable_count = category_counts.get("answerable", 0)
        answerable_ratio = answerable_count / total if total > 0 else 0.0

        report = {
            "total_samples": total,
            "category_distribution": category_counts,
            "answerable_ratio": round(answerable_ratio, 4),
            "empty_texts": empty_texts,
            "is_valid": True,
            "warnings": [],
        }

        warnings: list[str] = []

        if answerable_ratio < self.trainer_config.min_answerable_ratio:
            warnings.append(
                f"Answerable ratio {answerable_ratio:.2%} below minimum "
                f"{self.trainer_config.min_answerable_ratio:.2%}"
            )
            report["is_valid"] = False

        if empty_texts > 0:
            warnings.append(f"{empty_texts} examples have empty text")
            report["is_valid"] = False

        report["warnings"] = warnings

        if warnings:
            for warning in warnings:
                logger.warning("RAFT data validation warning", detail=warning)
        else:
            logger.info("RAFT data validation passed", **report)

        return report

    def prepare(self) -> None:
        """Prepare model, tokenizer, and training infrastructure.

        Loads the base model with quantization, optionally loads a pre-trained
        SFT adapter, and applies QLoRA adapters for RAFT training.

        Raises:
            RuntimeError: If required libraries are not installed.
        """
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "transformers is required. Install with: pip install transformers"
            )

        set_seed(self.trainer_config.seed)
        gpu_info = check_gpu_memory()

        logger.info(
            "Preparing RAFT trainer",
            model_id=self.trainer_config.model_id,
            sft_adapter=self.trainer_config.sft_adapter_path,
            quantization_bits=self.trainer_config.quantization_bits,
        )

        # Determine device map
        device_map = self.trainer_config.device_map
        if not gpu_info.get("gpu_available", False) and device_map == "auto":
            logger.warning("No GPU detected; falling back to CPU.")
            device_map = "cpu"

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.trainer_config.model_id,
            trust_remote_code=self.trainer_config.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"

        # Build quantization config
        bnb_config = None
        if self.trainer_config.quantization_bits > 0:
            try:
                bnb_config = get_bnb_config(bits=self.trainer_config.quantization_bits)
            except (RuntimeError, ImportError) as exc:
                logger.warning(
                    "Quantization unavailable, loading full precision",
                    error=str(exc),
                )

        # Load base model
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": self.trainer_config.trust_remote_code,
            "device_map": device_map,
        }
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config

        if TORCH_AVAILABLE and torch is not None:
            if self.trainer_config.bf16:
                model_kwargs["torch_dtype"] = torch.bfloat16
            elif self.trainer_config.fp16:
                model_kwargs["torch_dtype"] = torch.float16

        logger.info("Loading base model", model_id=self.trainer_config.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.trainer_config.model_id,
            **model_kwargs,
        )

        # Load SFT adapter if provided (two-stage: SFT → RAFT)
        if self.trainer_config.sft_adapter_path:
            self._load_sft_adapter()

        # Apply QLoRA for RAFT training
        if self.trainer_config.quantization_bits > 0:
            self.model = prepare_model_for_qlora(
                self.model,
                self.qlora_config,
                use_gradient_checkpointing=self.trainer_config.gradient_checkpointing,
            )
        else:
            logger.info("Skipping QLoRA preparation (quantization disabled)")

        # Save training config
        log_training_config(
            config=self.trainer_config.model_dump(),
            output_dir=self.trainer_config.output_dir,
        )

        self._is_prepared = True
        logger.info("RAFT trainer preparation complete")

    def _load_sft_adapter(self) -> None:
        """Load a pre-trained SFT LoRA adapter onto the base model.

        This enables the two-stage training pipeline where the model first
        undergoes SFT, then RAFT fine-tuning builds upon the SFT weights.

        Raises:
            RuntimeError: If peft is not installed.
            FileNotFoundError: If adapter path does not exist.
        """
        if not PEFT_AVAILABLE or PeftModel is None:
            raise RuntimeError(
                "peft is required to load SFT adapter. Install with: pip install peft"
            )

        adapter_path = self.trainer_config.sft_adapter_path
        if adapter_path is None:
            return

        if not Path(adapter_path).exists():
            raise FileNotFoundError(
                f"SFT adapter path does not exist: {adapter_path}"
            )

        logger.info("Loading SFT adapter for two-stage training", path=adapter_path)
        self.model = PeftModel.from_pretrained(
            self.model,
            adapter_path,
            is_trainable=True,
        )

        # Merge and unload to get a clean base for new LoRA layers
        logger.info("Merging SFT adapter into base model")
        self.model = self.model.merge_and_unload()
        logger.info("SFT adapter merged successfully")

    def _setup_mlflow(self) -> None:
        """Initialize MLflow tracking if available and configured."""
        if not MLFLOW_AVAILABLE or mlflow is None:
            return

        if self.trainer_config.mlflow_experiment is None:
            return

        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "./experiments/mlruns")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self.trainer_config.mlflow_experiment)

        logger.info(
            "MLflow configured for RAFT",
            experiment=self.trainer_config.mlflow_experiment,
            tracking_uri=tracking_uri,
        )

    def train(
        self,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
        skip_validation: bool = False,
    ) -> dict[str, Any]:
        """Run RAFT training loop.

        Args:
            train_dataset: HuggingFace Dataset with 'text' column.
            eval_dataset: Optional evaluation dataset.
            skip_validation: Skip data mix validation if True.

        Returns:
            Training metrics dictionary.

        Raises:
            RuntimeError: If prepare() hasn't been called or TRL is unavailable.
            ValueError: If data validation fails and skip_validation is False.
        """
        if not self._is_prepared:
            raise RuntimeError("Call prepare() before train()")

        if not TRL_AVAILABLE or SFTTrainer is None:
            raise RuntimeError(
                "trl is required for RAFT training. Install with: pip install trl"
            )

        # Validate training data mix
        if not skip_validation:
            validation_report = self._validate_training_data(train_dataset)
            if not validation_report["is_valid"]:
                warnings_text = "; ".join(validation_report.get("warnings", []))
                raise ValueError(
                    f"RAFT training data validation failed: {warnings_text}"
                )

        self._setup_mlflow()

        # Create training arguments
        training_args = create_training_args_from_config(
            config=self.trainer_config.model_dump(),
            output_dir=self.trainer_config.output_dir,
            run_name=f"medfusion-raft-{self.trainer_config.seed}",
        )

        logger.info(
            "Starting RAFT training",
            train_examples=len(train_dataset) if hasattr(train_dataset, "__len__") else "unknown",
            learning_rate=self.trainer_config.learning_rate,
        )

        # Build SFT trainer (TRL's SFTTrainer works for RAFT data format too)
        trainer_kwargs: dict[str, Any] = {
            "model": self.model,
            "args": training_args,
            "train_dataset": train_dataset,
            "processing_class": self.tokenizer,
        }

        if eval_dataset is not None:
            trainer_kwargs["eval_dataset"] = eval_dataset

        self.trainer = SFTTrainer(**trainer_kwargs)

        # Run training
        train_result = self.trainer.train()

        # Extract metrics
        metrics = {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "train_samples_per_second": train_result.metrics.get(
                "train_samples_per_second", 0
            ),
            "train_steps_per_second": train_result.metrics.get(
                "train_steps_per_second", 0
            ),
            "total_steps": train_result.global_step,
            "training_stage": "raft",
        }

        # Log to MLflow
        if MLFLOW_AVAILABLE and mlflow is not None and self.trainer_config.mlflow_experiment:
            try:
                with mlflow.start_run(nested=True):
                    mlflow.log_params(self.trainer_config.model_dump())
                    mlflow.log_metrics(metrics)
            except Exception as exc:
                logger.warning("MLflow logging failed", error=str(exc))

        logger.info("RAFT training completed", **metrics)
        return metrics

    def evaluate(self, eval_dataset: Optional[Any] = None) -> dict[str, Any]:
        """Run evaluation on a dataset.

        Args:
            eval_dataset: Dataset to evaluate. Uses trainer's eval_dataset if None.

        Returns:
            Evaluation metrics dictionary.

        Raises:
            RuntimeError: If trainer has not been initialized.
        """
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized. Call train() first.")

        logger.info("Running RAFT evaluation")
        eval_kwargs: dict[str, Any] = {}
        if eval_dataset is not None:
            eval_kwargs["eval_dataset"] = eval_dataset

        eval_result = self.trainer.evaluate(**eval_kwargs)
        logger.info("RAFT evaluation completed", **eval_result)
        return eval_result

    def save_adapter(
        self,
        output_dir: Optional[str] = None,
        adapter_name: str = "medfusion-raft-adapter",
    ) -> Path:
        """Save the RAFT LoRA adapter weights.

        Args:
            output_dir: Directory for saving adapter.
            adapter_name: Descriptive name for the adapter directory.

        Returns:
            Path to the saved adapter directory.

        Raises:
            RuntimeError: If model has not been prepared.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call prepare() first.")

        save_path = Path(output_dir or self.trainer_config.output_dir) / adapter_name
        save_path.mkdir(parents=True, exist_ok=True)

        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(str(save_path))
            logger.info("RAFT LoRA adapter saved", path=str(save_path))

        if self.tokenizer is not None and hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(str(save_path))
            logger.info("Tokenizer saved alongside RAFT adapter", path=str(save_path))

        return save_path

    def cleanup(self) -> None:
        """Release model and trainer resources to free GPU memory."""
        if TORCH_AVAILABLE and torch is not None and torch.cuda.is_available():
            if self.model is not None:
                del self.model
                self.model = None
            if self.trainer is not None:
                del self.trainer
                self.trainer = None
            torch.cuda.empty_cache()
            logger.info("GPU memory cleared after RAFT cleanup")
        else:
            self.model = None
            self.trainer = None
            logger.info("RAFT trainer resources released")

        self._is_prepared = False
