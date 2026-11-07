"""Training utilities for MedFusion-RAFT.

Provides hardware detection, seed management, training argument construction,
and configuration logging helpers used across SFT and RAFT training pipelines.
"""

from __future__ import annotations

import json
import os
import platform
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Optional dependency imports
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore[assignment]
    NUMPY_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

try:
    from transformers import TrainingArguments

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TrainingArguments = None  # type: ignore[assignment]
    TRANSFORMERS_AVAILABLE = False


def set_seed(seed: int = 42) -> None:
    """Set random seeds across all libraries for reproducibility.

    Sets seeds for Python's random module, NumPy, and PyTorch (CPU and CUDA).

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if NUMPY_AVAILABLE and np is not None:
        np.random.seed(seed)

    if TORCH_AVAILABLE and torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # Deterministic algorithms (may reduce performance)
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]

    logger.info("Random seeds set", seed=seed)


def check_gpu_memory() -> dict[str, Any]:
    """Check available GPU memory and utilization.

    Returns:
        Dictionary containing GPU memory information per device, or
        CPU-only status if no GPU is available.
    """
    result: dict[str, Any] = {
        "gpu_available": False,
        "num_gpus": 0,
        "devices": [],
    }

    if not TORCH_AVAILABLE or torch is None:
        result["error"] = "PyTorch not installed"
        return result

    if not torch.cuda.is_available():
        result["note"] = "No CUDA-capable GPU detected"
        return result

    result["gpu_available"] = True
    num_gpus = torch.cuda.device_count()
    result["num_gpus"] = num_gpus

    devices: list[dict[str, Any]] = []
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        total_mem = props.total_mem / (1024 ** 3)  # Convert to GB

        try:
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
            free = total_mem - allocated
        except RuntimeError:
            allocated = 0.0
            reserved = 0.0
            free = total_mem

        device_info = {
            "device_id": i,
            "name": props.name,
            "total_memory_gb": round(total_mem, 2),
            "allocated_memory_gb": round(allocated, 2),
            "reserved_memory_gb": round(reserved, 2),
            "free_memory_gb": round(free, 2),
            "compute_capability": f"{props.major}.{props.minor}",
            "multi_processor_count": props.multi_processor_count,
        }
        devices.append(device_info)

    result["devices"] = devices
    logger.info("GPU memory check completed", **result)
    return result


def get_hardware_info() -> dict[str, Any]:
    """Collect comprehensive hardware and software environment information.

    Returns:
        Dictionary with CPU, GPU, OS, Python, and library version details.
    """
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # PyTorch version and GPU info
    if TORCH_AVAILABLE and torch is not None:
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["cudnn_version"] = str(torch.backends.cudnn.version())  # type: ignore[attr-defined]
            info["gpu_count"] = torch.cuda.device_count()
            gpu_info = check_gpu_memory()
            info["gpu_devices"] = gpu_info.get("devices", [])
        else:
            info["cuda_version"] = None
            info["gpu_count"] = 0
    else:
        info["torch_version"] = None
        info["cuda_available"] = False

    # Transformers version
    try:
        import transformers
        info["transformers_version"] = transformers.__version__
    except ImportError:
        info["transformers_version"] = None

    # PEFT version
    try:
        import peft
        info["peft_version"] = peft.__version__
    except ImportError:
        info["peft_version"] = None

    # TRL version
    try:
        import trl
        info["trl_version"] = trl.__version__
    except ImportError:
        info["trl_version"] = None

    # bitsandbytes version
    try:
        import bitsandbytes
        info["bitsandbytes_version"] = bitsandbytes.__version__
    except ImportError:
        info["bitsandbytes_version"] = None

    logger.info("Hardware info collected", **{
        k: v for k, v in info.items() if k != "gpu_devices"
    })
    return info


def create_training_args_from_config(
    config: dict[str, Any],
    output_dir: Optional[str] = None,
    run_name: Optional[str] = None,
) -> Any:
    """Create HuggingFace TrainingArguments from a config dictionary.

    Translates a flat or nested configuration dictionary into a
    transformers.TrainingArguments instance with sensible defaults.

    Args:
        config: Training configuration dictionary (e.g., from training.yaml).
        output_dir: Override for output directory path.
        run_name: Optional MLflow/W&B run name.

    Returns:
        transformers.TrainingArguments instance.

    Raises:
        RuntimeError: If transformers is not installed.
    """
    if not TRANSFORMERS_AVAILABLE or TrainingArguments is None:
        raise RuntimeError(
            "The 'transformers' library is required to create TrainingArguments."
        )

    effective_output_dir = output_dir or config.get(
        "output_dir", "./experiments/checkpoints"
    )

    args_dict: dict[str, Any] = {
        "output_dir": effective_output_dir,
        "num_train_epochs": config.get("num_train_epochs", 2),
        "per_device_train_batch_size": config.get("per_device_train_batch_size", 1),
        "per_device_eval_batch_size": config.get("per_device_eval_batch_size", 1),
        "gradient_accumulation_steps": config.get("gradient_accumulation_steps", 8),
        "learning_rate": config.get("learning_rate", 2e-5),
        "weight_decay": config.get("weight_decay", 0.01),
        "warmup_ratio": config.get("warmup_ratio", 0.1),
        "lr_scheduler_type": config.get("lr_scheduler_type", "cosine"),
        "logging_steps": config.get("logging_steps", 10),
        "save_steps": config.get("save_steps", 100),
        "eval_steps": config.get("eval_steps", 50),
        "save_total_limit": config.get("save_total_limit", 3),
        "fp16": config.get("fp16", False),
        "bf16": config.get("bf16", True),
        "seed": config.get("seed", 42),
        "gradient_checkpointing": config.get("gradient_checkpointing", True),
        "optim": config.get("optimizer", "paged_adamw_8bit"),
        "evaluation_strategy": config.get("evaluation_strategy", "steps"),
        "save_strategy": config.get("save_strategy", "steps"),
        "logging_first_step": True,
        "remove_unused_columns": False,
        "report_to": config.get("report_to", "none"),
    }

    if run_name:
        args_dict["run_name"] = run_name

    # Handle gradient_checkpointing_kwargs
    if args_dict.get("gradient_checkpointing"):
        args_dict["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    # Remove None values
    args_dict = {k: v for k, v in args_dict.items() if v is not None}

    logger.info(
        "Created TrainingArguments",
        output_dir=effective_output_dir,
        epochs=args_dict.get("num_train_epochs"),
        lr=args_dict.get("learning_rate"),
        batch_size=args_dict.get("per_device_train_batch_size"),
    )

    return TrainingArguments(**args_dict)


def log_training_config(
    config: dict[str, Any],
    output_dir: str,
    filename: str = "training_config.json",
) -> Path:
    """Save training configuration as a JSON file for experiment reproducibility.

    Args:
        config: Training configuration dictionary.
        output_dir: Directory to save the configuration file.
        filename: Name of the output JSON file.

    Returns:
        Path to the saved configuration file.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    config_file = out_path / filename

    # Add metadata
    save_data = {
        "config": config,
        "hardware": get_hardware_info(),
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, default=str)

    logger.info("Training config saved", path=str(config_file))
    return config_file


def estimate_training_time(
    num_examples: int,
    num_epochs: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    seconds_per_step: float = 2.0,
) -> dict[str, Any]:
    """Estimate total training time based on dataset size and hyperparameters.

    Args:
        num_examples: Number of training examples.
        num_epochs: Number of training epochs.
        batch_size: Per-device batch size.
        gradient_accumulation_steps: Gradient accumulation steps.
        seconds_per_step: Estimated seconds per optimization step.

    Returns:
        Dictionary with step counts and time estimates.
    """
    effective_batch_size = batch_size * gradient_accumulation_steps
    steps_per_epoch = max(1, num_examples // effective_batch_size)
    total_steps = steps_per_epoch * num_epochs
    total_seconds = total_steps * seconds_per_step

    hours = total_seconds / 3600
    minutes = total_seconds / 60

    estimate = {
        "num_examples": num_examples,
        "num_epochs": num_epochs,
        "effective_batch_size": effective_batch_size,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "estimated_seconds": round(total_seconds, 1),
        "estimated_minutes": round(minutes, 1),
        "estimated_hours": round(hours, 2),
    }

    logger.info("Training time estimate", **estimate)
    return estimate
