"""CLI evaluation script for benchmark experiments in MedFusion-RAFT.

Loads experiment configurations, initializes the pipeline orchestrator,
evaluates on test benchmark datasets, and exports comprehensive metrics reports.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.runner import EvaluationRunner
from src.models.schemas import (
    AnswerabilityLabel,
    DatasetExample,
    ExperimentConfig,
    QuestionDifficulty,
)
from src.utils.config import load_app_config
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def create_synthetic_benchmark_dataset() -> list[DatasetExample]:
    """Generate representative clinical QA benchmark examples for testing and fallback."""
    return [
        DatasetExample(
            question_id="MED-001",
            question="What is the first-line pharmacotherapy for uncomplicated essential hypertension?",
            reference_answer="First-line pharmacotherapy for essential hypertension includes thiazide diuretics, ACE inhibitors, ARBs, and calcium channel blockers.",
            gold_passage="Clinical practice guidelines recommend thiazide diuretics, calcium channel blockers, and ACE inhibitors or ARBs as initial first-line therapy for essential hypertension.",
            gold_passage_id="P_HTN_01",
            doc_id="ACC_AHA_2017_Guidelines",
            specialty="Cardiology",
            difficulty=QuestionDifficulty.EASY,
            answerability=AnswerabilityLabel.ANSWERABLE,
        ),
        DatasetExample(
            question_id="MED-002",
            question="What are the absolute contraindications to fibrinolytic therapy in acute ischemic stroke?",
            reference_answer="Absolute contraindications include previous intracranial hemorrhage, known cerebrovascular vascular malformation, active internal bleeding, and recent intracranial surgery within 3 months.",
            gold_passage="Intravenous rtPA is contraindicated in active internal bleeding, prior intracranial hemorrhage, severe uncontrolled hypertension >185/110 mmHg, and intracranial neoplasm.",
            gold_passage_id="P_STROKE_02",
            doc_id="AHA_ASA_Stroke_2019",
            specialty="Neurology",
            difficulty=QuestionDifficulty.MEDIUM,
            answerability=AnswerabilityLabel.ANSWERABLE,
        ),
        DatasetExample(
            question_id="MED-003",
            question="What is the recommended dosing protocol for an unapproved synthetic peptide XYZ-99?",
            reference_answer="XYZ-99 has no FDA approval or established dosing protocols; evidence is unverified.",
            gold_passage=None,
            gold_passage_id=None,
            doc_id=None,
            specialty="Pharmacology",
            difficulty=QuestionDifficulty.HARD,
            answerability=AnswerabilityLabel.UNANSWERABLE,
        ),
    ]


def load_dataset_from_file(data_path: Path, allow_synthetic: bool = False) -> list[DatasetExample]:
    """Load benchmark dataset from JSON or JSONL file.

    Args:
        data_path: Path to dataset file.
        allow_synthetic: If True, allows synthetic fallback data for testing.

    Returns:
        List of parsed DatasetExample objects.

    Raises:
        FileNotFoundError: If dataset file does not exist and synthetic data not allowed.
        ValueError: If dataset file is empty or malformed.
    """
    if not data_path.is_file():
        if allow_synthetic:
            logger.warning(
                "Dataset file not found at %s. Using synthetic benchmark set (--smoke-test mode).",
                data_path,
            )
            return create_synthetic_benchmark_dataset()
        else:
            raise FileNotFoundError(
                f"Evaluation dataset not found: {data_path}\n\n"
                f"Please prepare your evaluation dataset first:\n"
                f"  python scripts/prepare_data.py --dataset <name> --output-dir data/processed\n\n"
                f"Or use --smoke-test flag for synthetic test data:\n"
                f"  python scripts/evaluate.py --smoke-test --experiment-id {data_path.stem}"
            )

    examples: list[DatasetExample] = []
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            if str(data_path).endswith(".jsonl"):
                for line in f:
                    line = line.strip()
                    if line:
                        examples.append(DatasetExample(**json.loads(line)))
            else:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        examples.append(DatasetExample(**item))
                elif isinstance(data, dict) and "examples" in data:
                    for item in data["examples"]:
                        examples.append(DatasetExample(**item))
        
        if not examples:
            if allow_synthetic:
                logger.warning(
                    "Dataset file %s is empty. Using synthetic benchmark set (--smoke-test mode).",
                    data_path,
                )
                return create_synthetic_benchmark_dataset()
            else:
                raise ValueError(
                    f"Evaluation dataset is empty: {data_path}\n\n"
                    f"Please ensure your dataset contains valid examples."
                )
        
        logger.info("Loaded %d benchmark examples from %s", len(examples), data_path)
        return examples
        
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        if allow_synthetic:
            logger.error(
                "Failed to parse dataset from %s: %s. Using synthetic benchmark set (--smoke-test mode).",
                data_path,
                exc,
            )
            return create_synthetic_benchmark_dataset()
        else:
            raise ValueError(
                f"Failed to parse evaluation dataset from {data_path}: {exc}\n\n"
                f"Please ensure your dataset is valid JSON or JSONL format with required fields."
            ) from exc


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for evaluation execution."""
    parser = argparse.ArgumentParser(
        description="MedFusion-RAFT Benchmark Evaluation Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default="E1_base",
        help="Experiment identifier from experiments.yaml (e.g., E1_base, E3_hybrid_rag, E9_full_system)",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="./configs",
        help="Directory containing YAML configuration files",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data/processed/test.jsonl",
        help="Path to evaluation benchmark dataset (JSON or JSONL)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./experiments/eval_results",
        help="Output directory to serialize metrics, CSV, and markdown report",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=False,
        help="Use mock providers for generator and verifier (recommended for CPU testing)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        default=False,
        help="Use synthetic test data instead of real evaluation dataset (for testing only)",
    )
    return parser.parse_args()


def main() -> int:
    """Execute main evaluation CLI routine."""
    setup_logging()
    args = parse_args()

    config_path = Path(args.config_dir).resolve()
    data_path = Path(args.data_path).resolve()
    output_dir = Path(args.output_dir).resolve()

    logger.info(
        "Initializing evaluation runner",
        experiment_id=args.experiment_id,
        config_dir=str(config_path),
        mock_mode=args.mock,
    )

    # 1. Load app configurations
    try:
        app_cfg = load_app_config(config_path)
    except Exception as exc:
        logger.error("Failed to load configs from %s: %s", config_path, exc)
        return 1

    exp_cfg = app_cfg.experiments.get(args.experiment_id)
    if exp_cfg is None:
        logger.warning(
            "Experiment ID '%s' not explicitly configured in experiments.yaml; building default.",
            args.experiment_id,
        )
        exp_cfg = ExperimentConfig(
            experiment_id=args.experiment_id,
            name=f"Evaluation {args.experiment_id}",
            generator="base",
            retrieval="hybrid" if "rag" in args.experiment_id or "hybrid" in args.experiment_id else None,
            verification="verification" in args.experiment_id or "full" in args.experiment_id,
            abstention="abstention" in args.experiment_id or "full" in args.experiment_id,
            routing="full" in args.experiment_id or "routing" in args.experiment_id,
        )

    # 2. Load dataset
    test_data = load_dataset_from_file(data_path, allow_synthetic=args.smoke_test)
    
    if args.smoke_test:
        logger.warning(
            "Running evaluation with synthetic test data (--smoke-test mode). "
            "Results are not representative of real performance."
        )

    # 3. Instantiate Runner
    runner = EvaluationRunner(
        config_dir=str(config_path),
        use_mock_models=args.mock,
    )

    # 4. Execute Benchmark
    logger.info("Executing benchmark for experiment '%s'...", exp_cfg.experiment_id)
    result = runner.run_experiment(exp_cfg, test_data)

    # 5. Export results and generate report
    runner.export_results(result, output_dir)
    report_file = output_dir / f"report_{args.experiment_id}.md"
    report_text = runner.generate_report(result, report_file)

    # 6. Terminal Summary Display
    print("\n" + "=" * 65)
    print(f" MEDFUSION-RAFT EVALUATION RESULT: {args.experiment_id}")
    print("=" * 65)
    print(f" Experiment Name       : {result.config.name}")
    print(f" Samples Evaluated     : {result.num_examples}")
    print(f" QA Accuracy           : {result.metrics.accuracy}")
    print(f" Token F1 Score        : {result.metrics.f1}")
    print(f" Claim Support Rate    : {result.metrics.claim_support_rate}")
    print(f" Selective Risk        : {result.metrics.selective_risk}")
    print(f" Answer Coverage       : {result.metrics.coverage}")
    print(f" ECE (Calibration)     : {result.metrics.ece}")
    print(f" Median Latency        : {result.metrics.median_latency_ms} ms")
    print(f" Report Saved To       : {report_file}")
    print("=" * 65 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
