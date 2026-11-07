#!/usr/bin/env python3
"""Supervised Fine-Tuning (SFT) training script for MedFusion-RAFT.

Usage:
    python scripts/train_sft.py --config configs/training.yaml --output-dir experiments/checkpoints/sft
    python scripts/train_sft.py --config configs/training.yaml --data-dir data/processed --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="MedFusion-RAFT SFT Training")
    parser.add_argument("--model-config", default="configs/models.yaml", help="Model configuration YAML")
    parser.add_argument("--training-config", default="configs/training.yaml", help="Training configuration YAML")
    parser.add_argument("--data-dir", default="data/processed", help="Directory with training data")
    parser.add_argument("--output-dir", default="experiments/checkpoints/sft", help="Output directory for checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Validate config without training")
    args = parser.parse_args()

    from src.training.utils import set_seed, get_hardware_info, check_gpu_memory, log_training_config

    # Set seed
    set_seed(args.seed)
    print(f"🌱 Random seed: {args.seed}")

    # Check hardware
    hw_info = get_hardware_info()
    print(f"🖥️  Hardware: {hw_info.get('gpu_name', 'CPU only')}")
    if hw_info.get("gpu_available"):
        mem = check_gpu_memory()
        print(f"💾 GPU Memory: {mem.get('free_mb', 0):.0f} MB free / {mem.get('total_mb', 0):.0f} MB total")

    # Load configs
    import yaml
    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)
    with open(args.training_config) as f:
        training_cfg = yaml.safe_load(f)

    sft_cfg = training_cfg.get("sft", {})
    qlora_cfg = training_cfg.get("qlora", {})

    print(f"\n📋 SFT Configuration:")
    print(f"   Learning rate: {sft_cfg.get('learning_rate', 2e-5)}")
    print(f"   Epochs: {sft_cfg.get('num_train_epochs', 2)}")
    print(f"   Batch size: {sft_cfg.get('per_device_train_batch_size', 1)}")
    print(f"   Gradient accumulation: {sft_cfg.get('gradient_accumulation_steps', 8)}")
    print(f"   Max seq length: {sft_cfg.get('max_seq_length', 2048)}")
    print(f"   QLoRA rank: {qlora_cfg.get('r', 16)}")

    if args.dry_run:
        print("\n✅ Dry run complete. Configuration is valid.")
        return

    # Check for training data
    data_dir = Path(args.data_dir)
    train_file = data_dir / "sft_train.jsonl"
    val_file = data_dir / "sft_val.jsonl"

    if not train_file.exists():
        print(f"\n⚠️  Training data not found at {train_file}")
        print("Run `python scripts/prepare_data.py` first to create training data.")
        print("\nAttempting to create sample training data for testing...")

        # Create minimal sample data for testing
        data_dir.mkdir(parents=True, exist_ok=True)
        sample_examples = [
            {
                "question": "What is the mechanism of action of metformin?",
                "context": "Metformin decreases hepatic glucose production and increases insulin sensitivity in peripheral tissues.",
                "answer": "Metformin works by decreasing hepatic glucose production and increasing insulin sensitivity in peripheral tissues [P1].",
                "citation_ids": ["P1"],
                "answerability": "answerable",
            },
            {
                "question": "What are the side effects of aspirin?",
                "context": "Common side effects of aspirin include gastrointestinal irritation, bleeding risk, and allergic reactions.",
                "answer": "Common side effects of aspirin include gastrointestinal irritation, increased bleeding risk, and allergic reactions [P1].",
                "citation_ids": ["P1"],
                "answerability": "answerable",
            },
        ]
        with open(train_file, "w") as f:
            for ex in sample_examples:
                f.write(json.dumps(ex) + "\n")
        with open(val_file, "w") as f:
            f.write(json.dumps(sample_examples[0]) + "\n")
        print(f"   Created sample data with {len(sample_examples)} examples")

    # Build SFT dataset
    print("\n📦 Building SFT dataset...")
    from src.training.sft.dataset import SFTDatasetBuilder, SFTDatasetConfig

    dataset_config = SFTDatasetConfig(
        max_seq_length=sft_cfg.get("max_seq_length", 2048),
    )
    builder = SFTDatasetBuilder(config=dataset_config)

    # Load training examples
    from src.models.schemas import TrainingSFTExample
    train_examples = []
    with open(train_file) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                train_examples.append(TrainingSFTExample(**data))

    val_examples = []
    if val_file.exists():
        with open(val_file) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    val_examples.append(TrainingSFTExample(**data))

    print(f"   Training examples: {len(train_examples)}")
    print(f"   Validation examples: {len(val_examples)}")

    train_dataset = builder.from_examples(train_examples)
    val_dataset = builder.from_examples(val_examples) if val_examples else None

    # Initialize trainer
    print("\n🏋️ Initializing SFT trainer...")
    try:
        from src.training.sft.trainer import MedFusionSFTTrainer
        from src.training.qlora.config import QLoRAConfig
        from src.models.base import ModelConfig

        gen_cfg = model_cfg.get("generator", {})
        model_config = ModelConfig(
            model_id=gen_cfg.get("model_id", "google/medgemma-1.5-4b-it"),
            revision=gen_cfg.get("revision", "main"),
            quantization_bits=gen_cfg.get("quantization", {}).get("bits", 4),
            quant_type=gen_cfg.get("quantization", {}).get("quant_type", "nf4"),
            compute_dtype=gen_cfg.get("quantization", {}).get("compute_dtype", "bfloat16"),
        )

        qlora_config = QLoRAConfig(
            r=qlora_cfg.get("r", 16),
            lora_alpha=qlora_cfg.get("lora_alpha", 32),
            lora_dropout=qlora_cfg.get("lora_dropout", 0.05),
        )

        trainer = MedFusionSFTTrainer(
            model_config=model_config,
            qlora_config=qlora_config,
            training_config=sft_cfg,
        )

        # Prepare
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        trainer.prepare(train_dataset, val_dataset)

        # Log config
        log_training_config(
            {
                "model": model_config.model_dump(),
                "qlora": qlora_config.model_dump(),
                "sft": sft_cfg,
                "seed": args.seed,
                "hardware": hw_info,
            },
            str(output_dir),
        )

        # Train
        print("\n🚀 Starting SFT training...")
        trainer.train()

        # Save adapter
        adapter_dir = output_dir / "best"
        trainer.save_adapter(str(adapter_dir))
        print(f"\n✅ Training complete. Adapter saved to {adapter_dir}")

    except ImportError as e:
        print(f"\n❌ Missing dependency for training: {e}")
        print("Install GPU dependencies: pip install torch transformers peft trl bitsandbytes accelerate")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

