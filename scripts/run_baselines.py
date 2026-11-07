#!/usr/bin/env python3
"""Run baseline experiments for MedFusion-RAFT.

Runs experiments E1-E3 (base, BM25 RAG, hybrid RAG) using mock or real models.

Usage:
    python scripts/run_baselines.py --mock --output-dir experiments/results
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Run MedFusion-RAFT baseline experiments")
    parser.add_argument("--mock", action="store_true", help="Use mock providers")
    parser.add_argument("--config-dir", default="configs", help="Config directory")
    parser.add_argument("--data-path", default="data/processed/test.jsonl", help="Test dataset")
    parser.add_argument("--output-dir", default="experiments/results", help="Output directory")
    parser.add_argument("--experiments", nargs="+", default=["E1_base", "E2_bm25_rag", "E3_hybrid_rag"],
                        help="Experiment IDs to run")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🔬 MedFusion-RAFT Baseline Experiments")
    print(f"   Mode: {'Mock' if args.mock else 'Real'}")
    print(f"   Experiments: {args.experiments}")
    print(f"   Output: {output_dir}\n")

    results = {}

    for exp_id in args.experiments:
        print(f"\n{'='*60}")
        print(f"Running {exp_id}...")
        print(f"{'='*60}")

        start = time.perf_counter()

        # For now, record experiment metadata
        result = {
            "experiment_id": exp_id,
            "status": "NOT_RUN" if not args.mock else "MOCK_RUN",
            "mode": "mock" if args.mock else "real",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metrics": {},
            "notes": "Baseline experiment — results from mock providers are for testing only, not research evaluation.",
        }

        if args.mock:
            # Run mock pipeline for each experiment config
            from scripts.run_pipeline import run_single_question
            sample_questions = [
                "What is the first-line treatment for hypertension?",
                "What are the contraindications for metformin?",
                "How is heart failure diagnosed?",
            ]

            question_results = []
            for q in sample_questions:
                qr = run_single_question(q, use_mock=True)
                question_results.append(qr)

            # Compute mock metrics
            n = len(question_results)
            result["metrics"] = {
                "num_questions": n,
                "accept_rate": sum(1 for r in question_results if r.get("decision") == "accept") / n,
                "abstain_rate": sum(1 for r in question_results if r.get("decision") == "abstain") / n,
                "avg_confidence": sum(r.get("confidence", 0) for r in question_results) / n,
                "avg_latency_ms": sum(r.get("total_time_ms", 0) for r in question_results) / n,
            }
            result["status"] = "MOCK_COMPLETED"
        else:
            result["notes"] = "Real model execution requires GPU and model access. Run with --mock for testing."

        elapsed = time.perf_counter() - start
        result["elapsed_seconds"] = elapsed
        result["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        results[exp_id] = result

        # Save individual result
        result_path = output_dir / f"{exp_id}.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"   Saved to {result_path}")

    # Save summary
    summary_path = output_dir / "baseline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✅ Baseline experiments complete. Summary: {summary_path}")


if __name__ == "__main__":
    main()

