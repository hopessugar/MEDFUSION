#!/usr/bin/env python3
"""Run MedFusion-RAFT on external medical QA benchmarks.

Downloads and evaluates on standard benchmarks including:
- MedQA (USMLE-style questions)
- MedMCQA (Indian medical exams)
- MMLU Medical (professional medicine)
- PubMedQA (biomedical research)

Usage:
    python scripts/run_external_benchmarks.py --benchmark medqa --max-samples 100
    python scripts/run_external_benchmarks.py --all --output experiments/benchmark_results/
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.external_benchmarks import BenchmarkRegistry
from src.evaluation.runner import EvaluationRunner
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run MedFusion-RAFT on external medical QA benchmarks"
    )
    
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["medqa", "medmcqa", "mmlu_medical", "pubmedqa", "all"],
        default="all",
        help="Benchmark to evaluate on"
    )
    
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use (train/validation/test)"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: all)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/benchmark_results",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--experiment-id",
        type=str,
        default="E9_full_system",
        help="Experiment configuration to use"
    )
    
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save individual predictions"
    )
    
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock models for testing"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def save_results(results: dict, output_path: Path):
    """Save benchmark results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")


def run_single_benchmark(
    benchmark_name: str,
    split: str,
    max_samples: int,
    experiment_id: str,
    output_dir: Path,
    save_predictions: bool,
    mock: bool
) -> dict:
    """Run evaluation on a single benchmark.
    
    Args:
        benchmark_name: Name of benchmark
        split: Dataset split
        max_samples: Maximum samples to evaluate
        experiment_id: Experiment configuration
        output_dir: Output directory
        save_predictions: Whether to save predictions
        mock: Whether to use mock models
        
    Returns:
        Results dictionary
    """
    logger.info(f"=" * 80)
    logger.info(f"Running benchmark: {benchmark_name}")
    logger.info(f"=" * 80)
    
    # Load benchmark
    registry = BenchmarkRegistry()
    
    try:
        logger.info(f"Loading {benchmark_name} dataset (split={split}, max_samples={max_samples or 'all'})...")
        examples = registry.load_benchmark(
            benchmark_name, 
            split=split,
            max_examples=max_samples
        )
        
        if not examples:
            logger.error(f"Failed to load {benchmark_name} or no examples found")
            return {"error": f"Failed to load {benchmark_name}"}
        
        logger.info(f"Loaded {len(examples)} examples from {benchmark_name}")
        
    except Exception as e:
        logger.error(f"Error loading {benchmark_name}: {e}", exc_info=True)
        return {"error": str(e)}
    
    # Save examples to temporary file for evaluation
    temp_data_path = output_dir / f"{benchmark_name}_{split}_temp.jsonl"
    temp_data_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(temp_data_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.model_dump()) + "\n")
    
    logger.info(f"Saved {len(examples)} examples to {temp_data_path}")
    
    # Run evaluation
    # Note: This requires the full pipeline to be set up
    # For now, we'll create a placeholder result structure
    
    start_time = time.time()
    
    # TODO: Integrate with actual evaluation runner
    # evaluator = EvaluationRunner(experiment_id=experiment_id, mock=mock)
    # results = evaluator.evaluate_dataset(temp_data_path)
    
    # Placeholder results
    results = {
        "benchmark": benchmark_name,
        "split": split,
        "experiment_id": experiment_id,
        "num_examples": len(examples),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": time.time() - start_time,
        "metrics": {
            "accuracy": 0.0,
            "f1": 0.0,
            "exact_match": 0.0,
            "note": "Evaluation requires full pipeline setup. This is a placeholder."
        },
        "metadata": {
            "benchmark_description": registry.benchmarks[benchmark_name].METADATA.description,
            "citation": registry.benchmarks[benchmark_name].METADATA.citation,
            "baseline_accuracy": registry.benchmarks[benchmark_name].METADATA.baseline_accuracy,
            "human_performance": registry.benchmarks[benchmark_name].METADATA.human_performance,
        }
    }
    
    # Save results
    result_path = output_dir / f"{benchmark_name}_{split}_results.json"
    save_results(results, result_path)
    
    # Print summary
    logger.info(f"\n{benchmark_name} Results:")
    logger.info(f"  Examples: {results['num_examples']}")
    logger.info(f"  Time: {results['elapsed_seconds']:.2f}s")
    logger.info(f"  Baseline: {results['metadata']['baseline_accuracy']}")
    logger.info(f"  Human: {results['metadata']['human_performance']}")
    
    return results


def run_all_benchmarks(args):
    """Run evaluation on all benchmarks."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    benchmarks = ["medqa", "medmcqa", "mmlu_medical", "pubmedqa"]
    
    all_results = {}
    
    for benchmark in benchmarks:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Benchmark {benchmarks.index(benchmark) + 1}/{len(benchmarks)}: {benchmark}")
        logger.info(f"{'=' * 80}\n")
        
        results = run_single_benchmark(
            benchmark_name=benchmark,
            split=args.split,
            max_samples=args.max_samples,
            experiment_id=args.experiment_id,
            output_dir=output_dir,
            save_predictions=args.save_predictions,
            mock=args.mock
        )
        
        all_results[benchmark] = results
    
    # Save aggregate results
    aggregate_path = output_dir / f"all_benchmarks_{args.split}_aggregate.json"
    save_results(all_results, aggregate_path)
    
    # Print comparison table
    logger.info(f"\n{'=' * 80}")
    logger.info("BENCHMARK COMPARISON")
    logger.info(f"{'=' * 80}\n")
    
    logger.info(f"{'Benchmark':<20} | {'Examples':>10} | {'Baseline':>10} | {'Human':>10}")
    logger.info("-" * 80)
    
    for benchmark, results in all_results.items():
        if "error" in results:
            logger.info(f"{benchmark:<20} | {'ERROR':>10} | {'-':>10} | {'-':>10}")
            continue
        
        metadata = results.get("metadata", {})
        num_examples = results.get("num_examples", 0)
        baseline = metadata.get("baseline_accuracy", 0.0)
        human = metadata.get("human_performance", 0.0)
        
        logger.info(
            f"{benchmark:<20} | {num_examples:>10} | {baseline:>9.1%} | {human:>9.1%}"
        )


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)
    
    logger.info("MedFusion-RAFT External Benchmark Evaluation")
    logger.info(f"Experiment: {args.experiment_id}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Mock mode: {args.mock}")
    
    # Run evaluation
    if args.benchmark == "all":
        run_all_benchmarks(args)
    else:
        run_single_benchmark(
            benchmark_name=args.benchmark,
            split=args.split,
            max_samples=args.max_samples,
            experiment_id=args.experiment_id,
            output_dir=Path(args.output_dir),
            save_predictions=args.save_predictions,
            mock=args.mock
        )
    
    logger.info("\nBenchmark evaluation complete!")
    logger.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
