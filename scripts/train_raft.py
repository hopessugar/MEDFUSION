#!/usr/bin/env python3
"""RAFT (Retrieval-Augmented Fine-Tuning) training script for MedFusion-RAFT.

Usage:
    python scripts/train_raft.py --config configs/training.yaml --output-dir experiments/checkpoints/raft
    python scripts/train_raft.py --config configs/training.yaml --sft-adapter experiments/checkpoints/sft/best
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="MedFusion-RAFT RAFT Training")
    parser.add_argument("--model-config", default="configs/models.yaml", help="Model configuration YAML")
    parser.add_argument("--training-config", default="configs/training.yaml", help="Training configuration YAML")
    parser.add_argument("--data-dir", default="data/processed", help="Directory with training data")
    parser.add_argument("--sft-adapter", default=None, help="Path to previous SFT adapter (for sequential SFT→RAFT)")
    parser.add_argument("--output-dir", default="experiments/checkpoints/raft", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Validate config without training")
    args = parser.parse_args()

    from src.training.utils import set_seed, get_hardware_info, check_gpu_memory, log_training_config

    set_seed(args.seed)
    print(f"🌱 Random seed: {args.seed}")

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

    raft_cfg = training_cfg.get("raft", {})
    qlora_cfg = training_cfg.get("qlora", {})

    print(f"\n📋 RAFT Configuration:")
    print(f"   Learning rate: {raft_cfg.get('learning_rate', 1e-5)}")
    print(f"   Epochs: {raft_cfg.get('num_train_epochs', 2)}")
    print(f"   Distractor ratio: {raft_cfg.get('distractor_ratio', 0.5)}")
    print(f"   Num distractor passages: {raft_cfg.get('num_distractor_passages', 3)}")
    if args.sft_adapter:
        print(f"   SFT adapter: {args.sft_adapter}")

    if args.dry_run:
        print("\n✅ Dry run complete. Configuration is valid.")
        return

    # Check for training data
    data_dir = Path(args.data_dir)
    train_file = data_dir / "raft_train.jsonl"
    val_file = data_dir / "raft_validation.jsonl"  # Fixed: correct filename

    if not train_file.exists():
        print(f"\n❌ ERROR: RAFT training data not found at {train_file}")
        print(f"   Run: python scripts/prepare_data.py --include-pubmedqa --build-raft --output-dir {data_dir}")
        print(f"   Or for testing: python scripts/prepare_data.py --smoke-test --build-raft --output-dir {data_dir}")
        sys.exit(1)
    
    if not val_file.exists():
        print(f"\n❌ ERROR: RAFT validation data not found at {val_file}")
        print(f"   Run: python scripts/prepare_data.py --include-pubmedqa --build-raft --output-dir {data_dir}")
        sys.exit(1)

    # Build RAFT dataset
    print("\n📦 Building RAFT dataset...")
    from src.training.raft.dataset import RAFTDatasetBuilder, RAFTDatasetConfig

    raft_dataset_config = RAFTDatasetConfig(
        distractor_ratio=raft_cfg.get("distractor_ratio", 0.5),
        num_distractor_passages=raft_cfg.get("num_distractor_passages", 3),
        max_seq_length=raft_cfg.get("max_seq_length", 2048),
    )
    builder = RAFTDatasetBuilder(config=raft_dataset_config)

    # Load examples
    from src.models.schemas import TrainingRAFTExample, Passage
    train_examples = []
    with open(train_file) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                # Convert passage dicts to Passage objects
                if "relevant_passages" in data:
                    data["relevant_passages"] = [
                        Passage(**p) if isinstance(p, dict) else p
                        for p in data["relevant_passages"]
                    ]
                if "distractor_passages" in data:
                    data["distractor_passages"] = [
                        Passage(**p) if isinstance(p, dict) else p
                        for p in data["distractor_passages"]
                    ]
                train_examples.append(TrainingRAFTExample(**data))

    print(f"   Training examples: {len(train_examples)}")
    
    # Load validation examples
    val_examples = []
    with open(val_file) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                # Convert passage dicts to Passage objects
                if "relevant_passages" in data:
                    data["relevant_passages"] = [
                        Passage(**p) if isinstance(p, dict) else p
                        for p in data["relevant_passages"]
                    ]
                if "distractor_passages" in data:
                    data["distractor_passages"] = [
                        Passage(**p) if isinstance(p, dict) else p
                        for p in data["distractor_passages"]
                    ]
                val_examples.append(TrainingRAFTExample(**data))
    
    print(f"   Validation examples: {len(val_examples)}")

    # Convert to HF datasets
    train_dataset = builder.to_hf_dataset(train_examples)
    val_dataset = builder.to_hf_dataset(val_examples) if val_examples else None

    # Initialize trainer
    print("\n🏋️ Initializing RAFT trainer...")
    try:
        from src.training.raft.trainer import MedFusionRAFTTrainer
        from src.training.qlora.config import QLoRAConfig
        from src.models.base import ModelConfig

        gen_cfg = model_cfg.get("generator", {})
        model_config = ModelConfig(
            model_id=gen_cfg.get("model_id", "google/medgemma-1.5-4b-it"),
            revision=gen_cfg.get("revision", "main"),
            quantization_bits=gen_cfg.get("quantization", {}).get("bits", 4),
            adapter_path=args.sft_adapter,
        )

        qlora_config = QLoRAConfig(
            r=qlora_cfg.get("r", 16),
            lora_alpha=qlora_cfg.get("lora_alpha", 32),
            lora_dropout=qlora_cfg.get("lora_dropout", 0.05),
        )

        trainer = MedFusionRAFTTrainer(
            model_config=model_config,
            qlora_config=qlora_config,
            training_config=raft_cfg,
        )

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        trainer.prepare(train_dataset, val_dataset)  # Fixed: pass validation dataset

        log_training_config(
            {
                "model": model_config.model_dump(),
                "qlora": qlora_config.model_dump(),
                "raft": raft_cfg,
                "seed": args.seed,
                "sft_adapter": args.sft_adapter,
                "hardware": hw_info,
            },
            str(output_dir),
        )

        print("\n🚀 Starting RAFT training...")
        trainer.train()

        adapter_dir = output_dir / "best"
        trainer.save_adapter(str(adapter_dir))
        print(f"\n✅ RAFT training complete. Adapter saved to {adapter_dir}")

    except ImportError as e:
        print(f"\n❌ Missing dependency: {e}")
        print("Install: pip install torch transformers peft trl bitsandbytes accelerate")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

