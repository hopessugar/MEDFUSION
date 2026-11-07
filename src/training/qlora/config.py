"""QLoRA configuration and utilities for parameter-efficient fine-tuning."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Optional dependencies handled gracefully
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

try:
    from transformers import BitsAndBytesConfig

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    BitsAndBytesConfig = None  # type: ignore[assignment]
    TRANSFORMERS_AVAILABLE = False

try:
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )

    PEFT_AVAILABLE = True
except ImportError:
    LoraConfig = None  # type: ignore[assignment]
    TaskType = None  # type: ignore[assignment]
    get_peft_model = None  # type: ignore[assignment]
    prepare_model_for_kbit_training = None  # type: ignore[assignment]
    PEFT_AVAILABLE = False


DEFAULT_GEMMA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


class QLoRAConfig(BaseModel):
    """Configuration model for QLoRA PEFT parameters.

    Attributes:
        r: LoRA attention dimension (rank). Default 16.
        lora_alpha: LoRA scaling factor. Default 32.
        lora_dropout: Dropout probability for LoRA layers. Default 0.05.
        bias: Bias type for LoRA ('none', 'all', 'lora_only'). Default 'none'.
        task_type: Target task architecture type. Default 'CAUSAL_LM'.
        target_modules: Explicit list of module names to inject LoRA into,
            or None to auto-detect target linear modules.
        modules_to_save: List of modules apart from adapter layers to be set as trainable.
    """

    r: int = Field(default=16, ge=1, description="LoRA rank")
    lora_alpha: int = Field(default=32, ge=1, description="LoRA alpha scaling factor")
    lora_dropout: float = Field(default=0.05, ge=0.0, le=1.0, description="LoRA dropout rate")
    bias: str = Field(default="none", description="Bias training strategy")
    task_type: str = Field(default="CAUSAL_LM", description="Task type for PEFT")
    target_modules: Optional[list[str]] = Field(
        default=None, description="Modules to adapt; None triggers auto-detection"
    )
    modules_to_save: Optional[list[str]] = Field(
        default=None, description="Extra module names to retain as trainable"
    )


def get_bnb_config(
    bits: int = 4,
    compute_dtype: str = "bfloat16",
    quant_type: str = "nf4",
    use_double_quant: bool = True,
) -> Optional[Any]:
    """Build BitsAndBytesConfig for model quantization.

    Args:
        bits: Quantization bit-width (4 or 8). 0 or None disables quantization.
        compute_dtype: Computation dtype string ('bfloat16', 'float16', 'float32').
        quant_type: 4-bit quantization datatype ('nf4' or 'fp4').
        use_double_quant: Whether to use nested quantization for 4-bit.

    Returns:
        Configured BitsAndBytesConfig or None if quantization is disabled.

    Raises:
        RuntimeError: If transformers is not installed.
        ValueError: If unsupported quantization bits are requested.
    """
    if not bits:
        return None

    if not TRANSFORMERS_AVAILABLE or BitsAndBytesConfig is None:
        raise RuntimeError("The transformers library is required to build BitsAndBytesConfig.")

    if not TORCH_AVAILABLE or torch is None:
        raise RuntimeError("PyTorch is required to build BitsAndBytesConfig.")

    resolved_dtype = getattr(torch, compute_dtype, torch.bfloat16)

    if bits == 4:
        logger.info(
            "Creating 4-bit BitsAndBytesConfig",
            quant_type=quant_type,
            compute_dtype=compute_dtype,
            use_double_quant=use_double_quant,
        )
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_type,
            bnb_4bit_compute_dtype=resolved_dtype,
            bnb_4bit_use_double_quant=use_double_quant,
        )
    elif bits == 8:
        logger.info("Creating 8-bit BitsAndBytesConfig")
        return BitsAndBytesConfig(load_in_8bit=True)

    raise ValueError(f"Unsupported quantization bit width: {bits}. Must be 4 or 8.")


def find_target_modules(model: Any) -> list[str]:
    """Auto-detect candidate linear layer names for LoRA injection.

    Args:
        model: PyTorch model instance.

    Returns:
        List of module class leaf names suitable for LoRA adapters.
    """
    if not TORCH_AVAILABLE or torch is None:
        return list(DEFAULT_GEMMA_TARGET_MODULES)

    cls_names: set[str] = set()
    for name, module in model.named_modules():
        module_type_str = str(type(module))
        if any(target in module_type_str for target in ["Linear", "Linear4bit", "Linear8bitLt"]):
            parts = name.split(".")
            cls_names.add(parts[-1])

    # Remove final head projection to avoid vocabulary parameter explosion
    for excluded in ["lm_head", "score", "embed_tokens"]:
        cls_names.discard(excluded)

    if not cls_names:
        logger.warning(
            "No standard linear layers detected automatically; defaulting to Gemma target modules."
        )
        return list(DEFAULT_GEMMA_TARGET_MODULES)

    detected = sorted(list(cls_names))
    logger.info("Auto-detected LoRA target modules", modules=detected)
    return detected


def get_lora_config(qlora_config: QLoRAConfig, model: Optional[Any] = None) -> Any:
    """Generate a PEFT LoraConfig instance from QLoRAConfig.

    Args:
        qlora_config: High-level Pydantic QLoRAConfig.
        model: Optional model instance used for auto-detecting target modules.

    Returns:
        peft.LoraConfig instance.

    Raises:
        RuntimeError: If peft is not installed.
    """
    if not PEFT_AVAILABLE or LoraConfig is None:
        raise RuntimeError("The 'peft' library is required to create a LoraConfig.")

    target_modules = qlora_config.target_modules
    if not target_modules:
        if model is not None:
            target_modules = find_target_modules(model)
        else:
            target_modules = list(DEFAULT_GEMMA_TARGET_MODULES)

    task_type_val = TaskType.CAUSAL_LM if hasattr(TaskType, "CAUSAL_LM") else "CAUSAL_LM"
    if hasattr(TaskType, qlora_config.task_type):
        task_type_val = getattr(TaskType, qlora_config.task_type)

    logger.info(
        "Configuring LoRA adapter",
        rank=qlora_config.r,
        alpha=qlora_config.lora_alpha,
        dropout=qlora_config.lora_dropout,
        target_modules=target_modules,
    )

    return LoraConfig(
        r=qlora_config.r,
        lora_alpha=qlora_config.lora_alpha,
        lora_dropout=qlora_config.lora_dropout,
        bias=qlora_config.bias,
        task_type=task_type_val,
        target_modules=target_modules,
        modules_to_save=qlora_config.modules_to_save,
    )


def prepare_model_for_qlora(
    model: Any,
    qlora_config: QLoRAConfig,
    use_gradient_checkpointing: bool = True,
) -> Any:
    """Prepare a quantized Hugging Face model for QLoRA fine-tuning.

    Args:
        model: Base quantized PyTorch causal LM.
        qlora_config: QLoRA configuration parameters.
        use_gradient_checkpointing: Whether to enable gradient checkpointing.

    Returns:
        PeftModel wrapped model with initialized LoRA adapters.

    Raises:
        RuntimeError: If peft is not installed.
    """
    if not PEFT_AVAILABLE or prepare_model_for_kbit_training is None or get_peft_model is None:
        raise RuntimeError("The 'peft' library is required to prepare model for QLoRA.")

    logger.info("Preparing quantized base model for k-bit training")
    prepared_model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )

    lora_config = get_lora_config(qlora_config, model=prepared_model)
    peft_model = get_peft_model(prepared_model, lora_config)

    param_stats = estimate_trainable_params(peft_model)
    logger.info("Initialized PEFT model with LoRA adapters", **param_stats)
    return peft_model


def estimate_trainable_params(model: Any) -> dict[str, Any]:
    """Calculate and return parameter counts and percentage of trainable parameters.

    Args:
        model: PyTorch model or PeftModel.

    Returns:
        Dictionary with keys:
            - total_params: Total parameter count.
            - trainable_params: Number of trainable parameters.
            - percent_trainable: Percentage of trainable parameters.
    """
    trainable_params = 0
    all_params = 0

    if hasattr(model, "parameters"):
        for param in model.parameters():
            numel = param.numel()
            all_params += numel
            if param.requires_grad:
                trainable_params += numel

    percent = (100.0 * trainable_params / all_params) if all_params > 0 else 0.0

    return {
        "total_params": all_params,
        "trainable_params": trainable_params,
        "percent_trainable": round(percent, 4),
    }
