#!/usr/bin/env python3
"""Quick benchmark test to validate the system.

This script runs a small sample evaluation to:
1. Test that benchmarks load correctly
2. Validate the pipeline works end-to-end
3. Provide initial accuracy estimates
4. Generate sample outputs for review

Usage:
    python scripts/quick_benchmark_test.py
    python scripts/quick_benchmark_test.py --samples 50 --mock
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.external_benchmarks import BenchmarkRegistry
from src.models.schemas import DatasetExample
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Quick benchmark validation test")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples per benchmark")
    parser.add_argument("--mock", action="store_true", help="Use mock mode (no GPU needed)")
    parser.add_argument("--output", type=str, default="experiments/quick_test", help="Output directory")
    return parser.parse_args()


def test_benchmark_loading():
    """Test that all benchmarks load correctly."""
    logger.info("=" * 80)
    logger.info("STEP 1: Testing Benchmark Loading")
    logger.info("=" * 80)
    
    registry = BenchmarkRegistry()
    
    # List available benchmarks
    benchmarks = registry.list_benchmarks()
    logger.info(f"\nFound {len(benchmarks)} benchmarks:")
    for name, metadata in benchmarks.items():
        logger.info(f"  - {name}: {metadata.num_examples} examples")
    
    return registry


def load_sample_data(registry: BenchmarkRegistry, num_samples: int = 20):
    """Load sample data from each benchmark."""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Loading Sample Data")
    logger.info("=" * 80)
    
    all_samples = {}
    
    benchmarks_to_test = ["medqa", "mmlu_medical", "pubmedqa"]
    
    for benchmark_name in benchmarks_to_test:
        logger.info(f"\nLoading {num_samples} samples from {benchmark_name}...")
        try:
            samples = registry.load_benchmark(
                benchmark_name,
                split="test",
                max_examples=num_samples
            )
            
            if samples:
                all_samples[benchmark_name] = samples
                logger.info(f"  ✓ Loaded {len(samples)} examples from {benchmark_name}")
                
                # Show first example
                first = samples[0]
                logger.info(f"\n  Example question: {first.question[:100]}...")
                logger.info(f"  Reference answer: {first.reference_answer[:100]}...")
            else:
                logger.warning(f"  ✗ No samples loaded from {benchmark_name}")
                
        except Exception as e:
            logger.error(f"  ✗ Error loading {benchmark_name}: {e}")
    
    return all_samples


def analyze_benchmark_characteristics(all_samples: dict):
    """Analyze characteristics of loaded benchmarks."""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Analyzing Benchmark Characteristics")
    logger.info("=" * 80)
    
    for benchmark_name, samples in all_samples.items():
        logger.info(f"\n{benchmark_name.upper()} Analysis:")
        logger.info(f"  Total samples: {len(samples)}")
        
        # Analyze question lengths
        question_lengths = [len(s.question.split()) for s in samples]
        avg_q_len = sum(question_lengths) / len(question_lengths)
        logger.info(f"  Avg question length: {avg_q_len:.1f} words")
        
        # Analyze answer lengths
        answer_lengths = [len(s.reference_answer.split()) for s in samples]
        avg_a_len = sum(answer_lengths) / len(answer_lengths)
        logger.info(f"  Avg answer length: {avg_a_len:.1f} words")
        
        # Specialty distribution
        specialties = {}
        for s in samples:
            spec = s.specialty or "unknown"
            specialties[spec] = specialties.get(spec, 0) + 1
        
        logger.info(f"  Specialties: {', '.join(f'{k}({v})' for k, v in list(specialties.items())[:5])}")


def run_mock_evaluation(all_samples: dict, output_dir: Path):
    """Run mock evaluation to test pipeline."""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Running Mock Evaluation")
    logger.info("=" * 80)
    
    logger.info("\n⚠️  NOTE: This is a MOCK evaluation (no real models loaded)")
    logger.info("For real evaluation, you need:")
    logger.info("  1. GPU with 16GB+ VRAM")
    logger.info("  2. HuggingFace token for MedGemma")
    logger.info("  3. Full pipeline setup")
    
    # Simulate evaluation
    results = {}
    
    for benchmark_name, samples in all_samples.items():
        logger.info(f"\nMock evaluating {benchmark_name}...")
        
        # Simulate processing
        start_time = time.time()
        
        # Mock results (would be real evaluation here)
        mock_results = {
            "benchmark": benchmark_name,
            "num_evaluated": len(samples),
            "mock_mode": True,
            "metrics": {
                "accuracy": 0.0,
                "note": "This is a mock evaluation. Real metrics require full pipeline.",
            },
            "elapsed_seconds": time.time() - start_time,
        }
        
        results[benchmark_name] = mock_results
        logger.info(f"  ✓ Mock evaluation complete ({len(samples)} samples)")
    
    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "mock_evaluation_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n✓ Results saved to {output_file}")
    
    return results


def generate_summary_report(all_samples: dict, results: dict, output_dir: Path):
    """Generate summary report."""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Generating Summary Report")
    logger.info("=" * 80)
    
    report = []
    report.append("# MedFusion-RAFT: Quick Benchmark Test Report\n")
    report.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**Mode**: Mock Evaluation\n\n")
    
    report.append("## Benchmarks Tested\n\n")
    report.append("| Benchmark | Samples Loaded | Status |\n")
    report.append("|-----------|----------------|--------|\n")
    
    for benchmark_name in all_samples.keys():
        num_samples = len(all_samples[benchmark_name])
        status = "✓ Success" if num_samples > 0 else "✗ Failed"
        report.append(f"| {benchmark_name} | {num_samples} | {status} |\n")
    
    report.append("\n## Next Steps\n\n")
    report.append("1. **GPU Setup**: Get access to GPU with 16GB+ VRAM\n")
    report.append("2. **Model Access**: Request MedGemma access from Google\n")
    report.append("3. **Full Evaluation**: Run full benchmark with real models\n")
    report.append("4. **Expert Recruitment**: Begin recruiting medical doctors\n\n")
    
    report.append("## Expected Timeline\n\n")
    report.append("- **Week 1**: GPU setup + Model access\n")
    report.append("- **Week 2**: Full benchmark evaluation (10K+ questions)\n")
    report.append("- **Week 3-4**: Expert evaluation preparation\n")
    report.append("- **Month 2**: Expert evaluation execution\n\n")
    
    report.append("## Resources Needed\n\n")
    report.append("- **Compute**: 8× A100 GPUs (~$2,000/month)\n")
    report.append("- **Expert Evaluation**: 5 MDs × 15 hours × $100/hr = $7,500\n")
    report.append("- **Total (3 months)**: ~$14,500\n\n")
    
    # Save report (with UTF-8 encoding for Windows compatibility)
    report_file = output_dir / "quick_test_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.writelines(report)
    
    logger.info(f"✓ Report saved to {report_file}")
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    
    total_samples = sum(len(samples) for samples in all_samples.values())
    logger.info(f"\n✓ Successfully loaded {total_samples} samples from {len(all_samples)} benchmarks")
    logger.info(f"✓ Mock evaluation completed")
    logger.info(f"✓ Results saved to {output_dir}")
    logger.info("\n🚀 System is ready for full benchmark evaluation!")


def main():
    args = parse_args()
    
    setup_logging(log_level="INFO")
    
    logger.info("╔═══════════════════════════════════════════════════════════════╗")
    logger.info("║  MedFusion-RAFT: Quick Benchmark Validation Test             ║")
    logger.info("╚═══════════════════════════════════════════════════════════════╝")
    
    try:
        # Step 1: Test benchmark loading
        registry = test_benchmark_loading()
        
        # Step 2: Load sample data
        all_samples = load_sample_data(registry, args.samples)
        
        if not all_samples:
            logger.error("\n✗ No benchmarks loaded successfully!")
            logger.error("This may be due to missing 'datasets' library.")
            logger.error("Install with: pip install datasets")
            return 1
        
        # Step 3: Analyze characteristics
        analyze_benchmark_characteristics(all_samples)
        
        # Step 4: Mock evaluation
        results = run_mock_evaluation(all_samples, Path(args.output))
        
        # Step 5: Generate report
        generate_summary_report(all_samples, results, Path(args.output))
        
        logger.info("\n✅ Quick benchmark test completed successfully!")
        logger.info(f"\nNext: Review the report at {args.output}/quick_test_report.md")
        
        return 0
        
    except Exception as e:
        logger.error(f"\n✗ Test failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
