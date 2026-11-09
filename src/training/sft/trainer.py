"""Supervised Fine-Tuning (SFT) trainer for MedFusion-RAFT.

Wraps TRL's SFTTrainer with QLoRA quantization support, MLflow logging,
and adapter-based model saving for parameter-efficient medical LM fine-tuning.
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


class SFTTrainerConfig(BaseModel):
    """Configuration for the MedFusion SFT trainer.

    Attributes:
        model_id: HuggingFace model identifier or local path.
        output_dir: Directory for checkpoints and outputs.
        learning_rate: Peak learning rate for optimizer.
        num_train_epochs: Number of training epochs.
        per_device_train_batch_size: Training batch size per GPU.
        gradient_accumulation_steps: Steps before optimizer update.
        warmup_ratio: Fraction of steps for LR warmup.
        weight_decay: AdamW weight decay coefficient.
        max_seq_length: Maximum sequence length for tokenization.
        optimizer: Optimizer name (e.g., 'paged_adamw_8bit').
        lr_scheduler_type: Learning rate scheduler type.
        gradient_checkpointing: Enable gradient checkpointing to save memory.
        logging_steps: Log metrics every N steps.
        save_steps: Save checkpoint every N steps.
        eval_steps: Evaluate every N steps.
        save_total_limit: Maximum number of checkpoints to keep.
        fp16: Use FP16 mixed precision.
        bf16: Use BF16 mixed precision.
        seed: Random seed.
        quantization_bits: Quantization bit-width (4 or 8). 0 disables.
        trust_remote_code: Whether to trust remote code from HF hub.
        mlflow_experiment: MLflow experiment name. None disables MLflow.
        device_map: Device placement strategy ('auto', 'cpu', etc.).
    """

    model_id: str = Field(
        default="google/medgemma-1.5-4b-it",
        description="HuggingFace model ID or local path",
    )
    output_dir: str = Field(
        default="./experiments/checkpoints/sft",
        description="Checkpoint output directory",
    )
    learning_rate: float = Field(default=2e-5, description="Peak learning rate")
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
    save_total_limit: int = Field(default=3, ge=1, description="Max saved checkpoints")
    fp16: bool = Field(default=False, description="Use FP16")
    bf16: bool = Field(default=True, description="Use BF16")
    seed: int = Field(default=42, description="Random seed")
    quantization_bits: int = Field(default=4, description="Quantization bits (0=disabled)")
    trust_remote_code: bool = Field(default=True, description="Trust remote code")
    mlflow_experiment: Optional[str] = Field(
        default=None, description="MLflow experiment name"
    )
    device_map: str = Field(default="auto", description="Device map strategy")


class MedFusionSFTTrainer:
    """SFT trainer for medical QA fine-tuning with QLoRA support.

    Manages the full training lifecycle: model loading with quantization,
    LoRA adapter injection, training, evaluation, and adapter saving.
    """

    def __init__(
        self,
        trainer_config: SFTTrainerConfig,
        qlora_config: Optional[QLoRAConfig] = None,
    ) -> None:
        """Initialize the MedFusion SFT trainer.

        Args:
            trainer_config: SFT training configuration.
            qlora_config: QLoRA adapter configuration. Uses defaults if None.
        """
        self.trainer_config = trainer_config
        self.qlora_config = qlora_config or QLoRAConfig()
        self.model: Any = None
        self.tokenizer: Any = None
        self.trainer: Any = None
        self._is_prepared = False

    def prepare(self) -> None:
        """Prepare model, tokenizer, and training infrastructure.

        Loads the base model with quantization, applies QLoRA adapters,
        and sets up the tokenizer. Must be called before train().

        Raises:
            RuntimeError: If required libraries are not installed.
        """
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "transformers is required. Install with: pip install transformers"
            )

        set_seed(self.trainer_config.seed)
        hw_info = get_hardware_info()
        gpu_info = check_gpu_memory()

        logger.info(
            "Preparing SFT trainer",
            model_id=self.trainer_config.model_id,
            quantization_bits=self.trainer_config.quantization_bits,
            gpu_available=gpu_info.get("gpu_available", False),
        )

        # Determine device map based on hardware
        device_map = self.trainer_config.device_map
        if not gpu_info.get("gpu_available", False) and device_map == "auto":
            logger.warning("No GPU detected; falling back to CPU. Training will be slow.")
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
                    "Quantization unavailable, loading in full precision",
                    error=str(exc),
                )

        # Load base model
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": self.trainer_config.trust_remote_code,
            "device_map": device_map,
        }
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config

        # Set torch dtype
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

        # Apply QLoRA adapters
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
        logger.info("SFT trainer preparation complete")

    def _setup_mlflow(self) -> None:
        """Initialize MLflow tracking if available and configured."""
        if not MLFLOW_AVAILABLE or mlflow is None:
            logger.info("MLflow not available; skipping experiment tracking")
            return

        if self.trainer_config.mlflow_experiment is None:
            logger.info("MLflow experiment not configured; skipping")
            return

        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "./experiments/mlruns")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self.trainer_config.mlflow_experiment)

        logger.info(
            "MLflow configured",
            experiment=self.trainer_config.mlflow_experiment,
            tracking_uri=tracking_uri,
        )

    def train(
        self,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
    ) -> dict[str, Any]:
        """Run SFT training loop.

        Args:
            train_dataset: HuggingFace Dataset with 'text' column for training.
            eval_dataset: Optional evaluation dataset with 'text' column.

        Returns:
            Training metrics dictionary.

        Raises:
            RuntimeError: If prepare() has not been called or TRL is unavailable.
        """
        if not self._is_prepared:
            raise RuntimeError("Call prepare() before train()")

        if not TRL_AVAILABLE or SFTTrainer is None:
            raise RuntimeError(
                "trl is required for SFT training. Install with: pip install trl"
            )

        self._setup_mlflow()

        # Create training arguments
        training_args = create_training_args_from_config(
            config=self.trainer_config.model_dump(),
            output_dir=self.trainer_config.output_dir,
            run_name=f"medfusion-sft-{self.trainer_config.seed}",
        )

        logger.info(
            "Starting SFT training",
            train_examples=len(train_dataset) if hasattr(train_dataset, "__len__") else "unknown",
            eval_examples=len(eval_dataset) if eval_dataset and hasattr(eval_dataset, "__len__") else 0,
        )

        # Build SFT trainer
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
        }

        # Log to MLflow
        if MLFLOW_AVAILABLE and mlflow is not None and self.trainer_config.mlflow_experiment:
            try:
                with mlflow.start_run(nested=True):
                    mlflow.log_params(self.trainer_config.model_dump())
                    mlflow.log_metrics(metrics)
            except Exception as exc:
                logger.warning("MLflow logging failed", error=str(exc))

        logger.info("SFT training completed", **metrics)
        return metrics

    def evaluate(self, eval_dataset: Optional[Any] = None) -> dict[str, Any]:
        """Run evaluation on a dataset.

        Args:
            eval_dataset: Dataset to evaluate. Uses trainer's eval_dataset if None.

        Returns:
            Evaluation metrics dictionary.

        Raises:
            RuntimeError: If trainer has not been initialized via train().
        """
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized. Call train() first.")

        logger.info("Running SFT evaluation")
        eval_kwargs: dict[str, Any] = {}
        if eval_dataset is not None:
            eval_kwargs["eval_dataset"] = eval_dataset

        eval_result = self.trainer.evaluate(**eval_kwargs)
        logger.info("SFT evaluation completed", **eval_result)
        return eval_result

    def save_adapter(
        self,
        output_dir: Optional[str] = None,
        adapter_name: str = "medfusion-sft-adapter",
    ) -> Path:
        """Save the LoRA adapter weights separately.

        Args:
            output_dir: Directory for saving adapter. Defaults to
                trainer_config.output_dir / adapter_name.
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

        # Save adapter weights
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(str(save_path))
            logger.info("LoRA adapter saved", path=str(save_path))
        else:
            logger.warning("Model does not support save_pretrained")

        # Save tokenizer alongside adapter
        if self.tokenizer is not None and hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(str(save_path))
            logger.info("Tokenizer saved alongside adapter", path=str(save_path))

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
            logger.info("GPU memory cleared after cleanup")
        else:
            self.model = None
            self.trainer = None
            logger.info("Trainer resources released")

        self._is_prepared = False
