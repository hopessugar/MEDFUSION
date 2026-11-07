"""Performance and resource profiling tracker for MedFusion-RAFT.

Measures stage-level latency percentiles (median, P95, P99), GPU memory utilization,
system RAM consumption, and query throughput during benchmark execution.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import psutil

from src.utils.logging import get_logger

logger = get_logger(__name__)


class PerformanceTracker:
    """Profiles latency distributions, memory usage, and throughput across pipeline stages."""

    def __init__(self) -> None:
        """Initialize performance profiling containers."""
        self._active_timers: dict[str, float] = {}
        self.stage_durations_ms: dict[str, list[float]] = defaultdict(list)
        self.gpu_memory_snapshots_mb: list[dict[str, float]] = []
        self.ram_memory_snapshots_mb: list[float] = []
        self._start_wall_time = time.perf_counter()

    def start_timer(self, stage: str) -> None:
        """Start a high-resolution timer for a designated pipeline stage.

        Args:
            stage: Stage identifier (e.g. 'retrieval', 'generation', 'verification').
        """
        self._active_timers[stage] = time.perf_counter()

    def stop_timer(self, stage: str) -> float:
        """Stop timer for the specified stage and record elapsed duration in ms.

        Args:
            stage: Stage identifier.

        Returns:
            Elapsed time in milliseconds.
        """
        start = self._active_timers.pop(stage, None)
        if start is None:
            logger.warning("stop_timer called for '%s' without corresponding start_timer", stage)
            return 0.0

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.stage_durations_ms[stage].append(elapsed_ms)
        return elapsed_ms

    def record_gpu_memory(self) -> dict[str, float]:
        """Capture a snapshot of GPU VRAM and host RAM utilization.

        Returns:
            Dictionary containing allocated and reserved memory metrics in MB.
        """
        snap: dict[str, float] = {
            "gpu_allocated_mb": 0.0,
            "gpu_reserved_mb": 0.0,
            "gpu_max_allocated_mb": 0.0,
        }

        try:
            import torch

            if torch.cuda.is_available():
                snap["gpu_allocated_mb"] = torch.cuda.memory_allocated() / (1024.0 * 1024.0)
                snap["gpu_reserved_mb"] = torch.cuda.memory_reserved() / (1024.0 * 1024.0)
                snap["gpu_max_allocated_mb"] = (
                    torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                )
        except ImportError:
            pass

        self.gpu_memory_snapshots_mb.append(snap)

        # Host RAM
        try:
            process = psutil.Process()
            ram_mb = process.memory_info().rss / (1024.0 * 1024.0)
            self.ram_memory_snapshots_mb.append(ram_mb)
        except Exception:
            pass

        return snap

    def get_latency_stats(self) -> dict[str, dict[str, float]]:
        """Calculate latency summary statistics (median, P95, P99, mean) per stage.

        Returns:
            Dictionary mapping stage names to latency percentiles in ms.
        """
        stats: dict[str, dict[str, float]] = {}

        for stage, durations in self.stage_durations_ms.items():
            if not durations:
                continue

            arr = np.asarray(durations, dtype=np.float64)
            stats[stage] = {
                "count": float(len(arr)),
                "mean_ms": round(float(np.mean(arr)), 2),
                "median_ms": round(float(np.median(arr)), 2),
                "p95_ms": round(float(np.percentile(arr, 95)), 2),
                "p99_ms": round(float(np.percentile(arr, 99)), 2),
                "min_ms": round(float(np.min(arr)), 2),
                "max_ms": round(float(np.max(arr)), 2),
            }

        return stats

    def get_memory_stats(self) -> dict[str, float]:
        """Summarize peak and average memory consumption during the run.

        Returns:
            Dictionary with peak and average GPU/RAM metrics.
        """
        peak_gpu_allocated = 0.0
        peak_gpu_reserved = 0.0
        avg_gpu_allocated = 0.0

        if self.gpu_memory_snapshots_mb:
            allocs = [s["gpu_allocated_mb"] for s in self.gpu_memory_snapshots_mb]
            reservs = [s["gpu_reserved_mb"] for s in self.gpu_memory_snapshots_mb]
            peak_gpu_allocated = max(allocs)
            peak_gpu_reserved = max(reservs)
            avg_gpu_allocated = float(np.mean(allocs))

        peak_ram = max(self.ram_memory_snapshots_mb) if self.ram_memory_snapshots_mb else 0.0
        avg_ram = (
            float(np.mean(self.ram_memory_snapshots_mb)) if self.ram_memory_snapshots_mb else 0.0
        )

        return {
            "peak_gpu_allocated_mb": round(peak_gpu_allocated, 2),
            "peak_gpu_reserved_mb": round(peak_gpu_reserved, 2),
            "avg_gpu_allocated_mb": round(avg_gpu_allocated, 2),
            "peak_ram_mb": round(peak_ram, 2),
            "avg_ram_mb": round(avg_ram, 2),
        }

    def get_throughput(self, total_time: float, num_examples: int) -> float:
        """Calculate system processing throughput.

        Args:
            total_time: Total elapsed duration in seconds.
            num_examples: Count of processed samples.

        Returns:
            Throughput rate in queries per second.
        """
        if total_time <= 0 or num_examples <= 0:
            return 0.0
        return round(num_examples / total_time, 3)

    def export_report(self, output_path: str) -> dict[str, Any]:
        """Export comprehensive latency, memory, and throughput metrics to JSON.

        Args:
            output_path: Target JSON file path.

        Returns:
            Aggregated report dictionary.
        """
        total_time_s = time.perf_counter() - self._start_wall_time
        total_queries = sum(len(d) for d in self.stage_durations_ms.values())

        report = {
            "latency": self.get_latency_stats(),
            "memory": self.get_memory_stats(),
            "throughput_qps": self.get_throughput(
                total_time_s, max(1, len(self.stage_durations_ms.get("total", [])))
            ),
            "total_wall_time_s": round(total_time_s, 2),
        }

        dest = Path(output_path).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        logger.info("Exported performance report to %s", dest)
        return report
