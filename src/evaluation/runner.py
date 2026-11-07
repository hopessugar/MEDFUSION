"""Evaluation runner and experiment benchmark harness for MedFusion-RAFT.

Coordinates dataset evaluation, ablation experiment suites, metric calculation,
statistical significance comparison, and markdown/JSON report generation.
"""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from src.evaluation.answer_metrics import evaluate_answers, normalize_text
from src.evaluation.calibration import (
    compute_brier_score,
    compute_ece,
    compute_selective_risk,
)
from src.evaluation.grounding_metrics import evaluate_grounding
from src.evaluation.performance import PerformanceTracker
from src.evaluation.statistical import paired_bootstrap_test
from src.models.schemas import (
    AnswerabilityLabel,
    DatasetExample,
    DecisionType,
    EvaluationMetrics,
    ExperimentConfig,
    ExperimentResult,
    MedicalQuestion,
    PipelineResult,
)
from src.pipeline.orchestrator import MedFusionOrchestrator, create_orchestrator_from_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


class EvaluationRunner:
    """End-to-end benchmark evaluation runner for MedFusion-RAFT models."""

    def __init__(
        self,
        orchestrator: Optional[MedFusionOrchestrator] = None,
        config_dir: str = "./configs",
        use_mock_models: bool = False,
    ) -> None:
        """Initialize evaluation runner.

        Args:
            orchestrator: Configured MedFusionOrchestrator instance.
            config_dir: Directory containing YAML system configurations.
            use_mock_models: Whether to force mock models for CPU testing.
        """
        self.orchestrator = orchestrator
        self.config_dir = config_dir
        self.use_mock_models = use_mock_models
        self.tracker = PerformanceTracker()

    def evaluate_dataset(
        self,
        test_data: list[DatasetExample],
        experiment_config: Optional[ExperimentConfig] = None,
    ) -> ExperimentResult:
        """Execute evaluation over a test dataset and calculate comprehensive metrics.

        Args:
            test_data: List of DatasetExample benchmark questions and references.
            experiment_config: Optional specification for metadata tagging.

        Returns:
            Populated ExperimentResult with computed EvaluationMetrics.
        """
        if self.orchestrator is None:
            raise RuntimeError("EvaluationRunner requires a configured MedFusionOrchestrator")

        start_time = datetime.now(timezone.utc)
        t0 = time.perf_counter()

        logger.info("Commencing dataset evaluation over %d examples", len(test_data))
        pipeline_results: list[PipelineResult] = []

        for ex in test_data:
            q = MedicalQuestion(
                question_id=ex.question_id,
                question=ex.question,
                specialty=ex.specialty,
                difficulty=ex.difficulty,
                expected_answerability=ex.answerability,
            )

            self.tracker.start_timer("total")
            res = self.orchestrator.process_question(q)
            self.tracker.stop_timer("total")
            self.tracker.record_gpu_memory()

            pipeline_results.append(res)

        total_elapsed_s = time.perf_counter() - t0

        # Answer quality metrics
        ans_metrics = evaluate_answers(pipeline_results, test_data)

        # Grounding metrics
        ground_metrics = evaluate_grounding(pipeline_results)

        # Correctness & abstention calibration metrics
        correctness: list[int] = []
        confidences: list[float] = []
        abstained_flags: list[bool] = []
        unanswerable_ground_truth: list[bool] = []

        for r, ex in zip(pipeline_results, test_data):
            p_text = r.final_answer or (
                r.generated_answer.answer_text if r.generated_answer else ""
            )
            r_text = ex.reference_answer

            is_correct = int(normalize_text(r_text) in normalize_text(p_text))
            correctness.append(is_correct)

            score = r.decision.score if r.decision else (
                r.generated_answer.confidence if r.generated_answer and r.generated_answer.confidence is not None else 1.0
            )
            confidences.append(score)

            is_abstained = bool(r.decision and r.decision.decision == DecisionType.ABSTAIN)
            abstained_flags.append(is_abstained)
            unanswerable_ground_truth.append(
                ex.answerability == AnswerabilityLabel.UNANSWERABLE
            )

        ece = compute_ece(confidences, correctness)
        brier = compute_brier_score(confidences, correctness)

        # Abstention precision / recall / selective risk
        n_samples = max(1, len(test_data))
        num_accepted = sum(1 for ab in abstained_flags if not ab)
        coverage = num_accepted / n_samples
        selective_risk = compute_selective_risk(confidences, correctness, coverage=max(0.01, coverage))

        # Precision & Recall on abstention vs unanswerable
        true_unans = sum(1 for u in unanswerable_ground_truth if u)
        pred_unans = sum(1 for ab in abstained_flags if ab)
        tp_unans = sum(
            1 for ab, u in zip(abstained_flags, unanswerable_ground_truth) if ab and u
        )

        abs_prec = (tp_unans / pred_unans) if pred_unans > 0 else 1.0
        abs_rec = (tp_unans / true_unans) if true_unans > 0 else 1.0

        # Latency & memory stats
        lat_stats = self.tracker.get_latency_stats()
        mem_stats = self.tracker.get_memory_stats()
        total_lat = lat_stats.get("total", {})

        metrics = EvaluationMetrics(
            accuracy=ans_metrics.get("accuracy"),
            f1=ans_metrics.get("f1"),
            exact_match=ans_metrics.get("exact_match"),
            claim_support_rate=ground_metrics.get("claim_support_rate"),
            citation_precision=ground_metrics.get("citation_precision"),
            unsupported_claim_rate=ground_metrics.get("unsupported_claim_rate"),
            contradiction_rate=ground_metrics.get("contradiction_rate"),
            abstention_precision=round(abs_prec, 4),
            abstention_recall=round(abs_rec, 4),
            coverage=round(coverage, 4),
            selective_risk=round(selective_risk, 4),
            ece=round(ece, 4),
            brier_score=round(brier, 4),
            median_latency_ms=total_lat.get("median_ms"),
            p95_latency_ms=total_lat.get("p95_ms"),
            peak_gpu_memory_mb=mem_stats.get("peak_gpu_allocated_mb"),
        )

        cfg = experiment_config or ExperimentConfig(
            experiment_id="eval_run",
            name="Evaluation Run",
            generator="base",
        )

        return ExperimentResult(
            config=cfg,
            metrics=metrics,
            num_examples=len(test_data),
            status="COMPLETED",
            started_at=start_time,
            completed_at=datetime.now(timezone.utc),
            hardware_info=mem_stats,
        )

    def run_experiment(
        self,
        experiment_config: ExperimentConfig,
        test_data: list[DatasetExample],
    ) -> ExperimentResult:
        """Run benchmark pipeline for a given experimental configuration.

        Args:
            experiment_config: Configuration defining generator, retrieval, verifier, etc.
            test_data: Evaluation dataset.

        Returns:
            Calculated ExperimentResult.
        """
        logger.info("Executing experiment %s: %s", experiment_config.experiment_id, experiment_config.name)
        self.orchestrator = create_orchestrator_from_config(
            config_dir=self.config_dir,
            experiment_id=experiment_config.experiment_id,
            use_mock_models=self.use_mock_models,
        )
        return self.evaluate_dataset(test_data, experiment_config=experiment_config)

    def run_ablation(
        self,
        base_config: ExperimentConfig,
        components_to_ablate: list[str],
        test_data: list[DatasetExample],
    ) -> list[ExperimentResult]:
        """Execute systematic ablation by selectively disabling pipeline components.

        Args:
            base_config: Reference complete system configuration.
            components_to_ablate: List of component names ('retrieval', 'verification', 'abstention', 'routing').
            test_data: Evaluation dataset.

        Returns:
            List of ExperimentResult instances corresponding to each ablation condition.
        """
        results: list[ExperimentResult] = []

        # 1. Run baseline
        logger.info("Running baseline for ablation study...")
        baseline_res = self.run_experiment(base_config, test_data)
        results.append(baseline_res)

        # 2. Run ablations
        for comp in components_to_ablate:
            ablate_dict = base_config.model_dump()
            ablate_dict["experiment_id"] = f"{base_config.experiment_id}_no_{comp}"
            ablate_dict["name"] = f"{base_config.name} (w/o {comp})"

            if comp == "retrieval":
                ablate_dict["retrieval"] = None
            elif comp == "verification":
                ablate_dict["verification"] = False
            elif comp == "abstention":
                ablate_dict["abstention"] = False
            elif comp == "routing":
                ablate_dict["routing"] = False

            ablated_cfg = ExperimentConfig(**ablate_dict)
            res = self.run_experiment(ablated_cfg, test_data)
            results.append(res)

        return results

    def compare_experiments(
        self, results: list[ExperimentResult]
    ) -> dict[str, Any]:
        """Compile comparative metrics summary across multiple experimental outcomes.

        Args:
            results: List of ExperimentResult items to compare.

        Returns:
            Structured comparative metrics dictionary.
        """
        summary_rows: list[dict[str, Any]] = []

        for r in results:
            summary_rows.append({
                "experiment_id": r.config.experiment_id,
                "name": r.config.name,
                "accuracy": r.metrics.accuracy,
                "f1": r.metrics.f1,
                "claim_support_rate": r.metrics.claim_support_rate,
                "selective_risk": r.metrics.selective_risk,
                "coverage": r.metrics.coverage,
                "ece": r.metrics.ece,
                "median_latency_ms": r.metrics.median_latency_ms,
            })

        return {"comparison_table": summary_rows}

    def export_results(
        self,
        results: Union[ExperimentResult, list[ExperimentResult]],
        output_dir: Union[str, Path],
    ) -> None:
        """Serialize experiment evaluation results to JSON and CSV summaries.

        Args:
            results: Single ExperimentResult or collection of results.
            output_dir: Destination folder path.
        """
        res_list = [results] if isinstance(results, ExperimentResult) else results
        out_path = Path(output_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        # 1. JSON dump
        json_path = out_path / "experiment_results.json"
        data = [r.model_dump(mode="json") for r in res_list]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        # 2. CSV summary
        csv_path = out_path / "experiment_summary.csv"
        comparison = self.compare_experiments(res_list)["comparison_table"]
        if comparison:
            keys = list(comparison[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(comparison)

        logger.info("Exported evaluation results to %s and %s", json_path, csv_path)

    def generate_report(
        self,
        results: Union[ExperimentResult, list[ExperimentResult]],
        output_path: Union[str, Path],
    ) -> str:
        """Generate a formatted markdown evaluation report with metrics tables.

        Args:
            results: Evaluated experiment results.
            output_path: Markdown report file destination.

        Returns:
            Generated report markdown string.
        """
        res_list = [results] if isinstance(results, ExperimentResult) else results
        rows = self.compare_experiments(res_list)["comparison_table"]

        lines: list[str] = [
            "# MedFusion-RAFT Benchmark Evaluation Report",
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            "## Summary Table",
            "",
            "| Experiment ID | Name | Accuracy | F1 | Claim Support | Selective Risk | Coverage | ECE | Median Latency (ms) |",
            "|---|---|---|---|---|---|---|---|---|",
        ]

        for r in rows:
            lines.append(
                f"| `{r['experiment_id']}` | {r['name']} | "
                f"{r['accuracy'] or 0.0:.3f} | {r['f1'] or 0.0:.3f} | "
                f"{r['claim_support_rate'] or 0.0:.3f} | {r['selective_risk'] or 0.0:.3f} | "
                f"{r['coverage'] or 0.0:.3f} | {r['ece'] or 0.0:.3f} | "
                f"{r['median_latency_ms'] or 0.0:.1f} |"
            )

        lines.extend([
            "",
            "## Methodology & Guardrails",
            "- **Accuracy**: Exact & token containment matching against gold reference answers.",
            "- **Claim Support Rate**: Ratio of extracted atomic propositions verified by evidence passages.",
            "- **Selective Risk**: Error rate evaluated specifically over non-abstained predictions.",
            "- **Expected Calibration Error (ECE)**: Agreement between model confidence and empirical accuracy.",
            "",
        ])

        report_content = "\n".join(lines)
        dest = Path(output_path).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            f.write(report_content)

        logger.info("Markdown benchmark report generated at %s", dest)
        return report_content
