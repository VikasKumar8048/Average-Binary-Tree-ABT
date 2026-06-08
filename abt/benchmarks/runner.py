"""
abt/benchmarks/runner.py
========================
Benchmark suite reproducing Table III from the paper.

Compares three approaches:
  A1. ABT (Algorithm 1) – single BFS pass, produces navigable tree
  A2. Segment-Tree Aggregation (STA) – bottom-up flat array
  A3. Array Pyramid (APY) – iterative halving of 1-D array

Measured metrics:
  - Wall-clock construction time (milliseconds)
  - Peak memory usage (kilobytes)
  - Node count / compression ratio
  - Scalability across heights h in {8, 10, 12, 14, 16}

Reference: Paper Section VI-C, Table III.
"""

from __future__ import annotations

import gc
import math
import random
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from abt.core import (
    Node,
    build_abt,
    build_tree_from_list,
    bfs_values,
    height as tree_height,
    node_count,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    method: str
    h: int
    n: int
    time_ms: float
    memory_kb: float
    output_nodes: int
    compression_ratio: float

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "h": self.h,
            "n": self.n,
            "time_ms": round(self.time_ms, 3),
            "memory_kb": round(self.memory_kb, 2),
            "output_nodes": self.output_nodes,
            "compression_ratio": round(self.compression_ratio, 4),
        }


@dataclass
class BenchmarkReport:
    results: List[BenchmarkResult] = field(default_factory=list)
    runs_per_case: int = 5

    def summary_table(self) -> str:
        """Return a Markdown-formatted summary table."""
        header = (
            "| Method | h | n | Time (ms) | Memory (KB) | Output nodes | Ratio |\n"
            "|--------|---|---|-----------|-------------|--------------|-------|\n"
        )
        rows = []
        for r in self.results:
            rows.append(
                f"| {r.method} | {r.h} | {r.n} | "
                f"{r.time_ms:.2f} | {r.memory_kb:.0f} | "
                f"{r.output_nodes} | {r.compression_ratio:.4f} |"
            )
        return header + "\n".join(rows)

    def to_csv(self) -> str:
        lines = ["method,h,n,time_ms,memory_kb,output_nodes,compression_ratio"]
        for r in self.results:
            lines.append(
                f"{r.method},{r.h},{r.n},"
                f"{r.time_ms:.3f},{r.memory_kb:.2f},"
                f"{r.output_nodes},{r.compression_ratio:.4f}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Method A1: ABT
# ---------------------------------------------------------------------------

def _run_abt(root: Node) -> int:
    """Build ABT and return output node count."""
    result = build_abt(root)
    return node_count(result)


# ---------------------------------------------------------------------------
# Method A2: Segment-Tree Aggregation (STA)
# ---------------------------------------------------------------------------

def _build_segment_tree(values: List[float]) -> List[float]:
    """Bottom-up segment tree construction.

    Stores per-node averages in a flat array of size 2*n.
    Indices: 1-based.  Node i has children 2i and 2i+1.

    Reference: Paper Section VI-C, A2.
    """
    n = len(values)
    seg = [0.0] * (2 * n)
    # Leaf layer
    for i, v in enumerate(values):
        seg[n + i] = v
    # Build upward
    for i in range(n - 1, 0, -1):
        seg[i] = (seg[2 * i] + seg[2 * i + 1]) / 2.0
    return seg


def _run_sta(values: List[float]) -> int:
    """Run STA and return output array length."""
    seg = _build_segment_tree(values)
    return len(seg)


# ---------------------------------------------------------------------------
# Method A3: Array Pyramid (APY)
# ---------------------------------------------------------------------------

def _build_array_pyramid(values: List[float]) -> List[List[float]]:
    """Iterative pairwise halving of a 1-D array.

    Each level is produced by averaging adjacent pairs.
    Output: list of log(n) arrays.

    Reference: Paper Section VI-C, A3.
    """
    levels: List[List[float]] = [list(values)]
    current = list(values)
    while len(current) > 1:
        next_level = [
            (current[i] + current[i + 1]) / 2.0
            for i in range(0, len(current) - 1, 2)
        ]
        levels.append(next_level)
        current = next_level
    return levels


def _run_apy(values: List[float]) -> int:
    """Run APY and return total elements across all levels."""
    pyramid = _build_array_pyramid(values)
    return sum(len(lvl) for lvl in pyramid)


# ---------------------------------------------------------------------------
# Timing and memory measurement
# ---------------------------------------------------------------------------

def _measure(fn, *args, runs: int = 5) -> Tuple[float, float, int]:
    """Measure average time (ms) and peak memory (KB) over *runs* calls.

    Returns (avg_time_ms, peak_memory_kb, last_output_nodes).
    """
    times = []
    peak_kb = 0.0
    out_nodes = 0
    for _ in range(runs):
        gc.collect()
        tracemalloc.start()
        t0 = time.perf_counter()
        out_nodes = fn(*args)
        t1 = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        times.append((t1 - t0) * 1000.0)
        peak_kb = max(peak_kb, peak / 1024.0)

    avg_ms = sum(times) / len(times)
    return avg_ms, peak_kb, out_nodes


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

class ABTBenchmark:
    """Reproduce the benchmarks of Table III in the paper.

    Parameters
    ----------
    heights : list of int
        Perfect binary tree heights to benchmark.
        Paper uses {8, 10, 12, 14, 16}.
    runs : int
        Number of repetitions per (method, height) combination.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        heights: Optional[List[int]] = None,
        runs: int = 5,
        seed: int = 42,
    ) -> None:
        self.heights = heights if heights is not None else [8, 10, 12, 14]
        self.runs = runs
        self.seed = seed

    def _make_tree(self, h: int) -> Tuple[Node, List[float], int]:
        """Generate a random perfect binary tree of height h."""
        rng = random.Random(self.seed)
        n = (1 << (h + 1)) - 1  # 2^(h+1) - 1
        values = [rng.uniform(0.0, 1000.0) for _ in range(n)]
        root = build_tree_from_list(values)
        return root, values, n

    def run(self) -> BenchmarkReport:
        """Execute the full benchmark suite.

        Returns a BenchmarkReport with all results.
        """
        report = BenchmarkReport(runs_per_case=self.runs)

        for h in self.heights:
            root, values, n = self._make_tree(h)
            print(f"  h={h:2d}  n={n:6d}", end="  ", flush=True)

            # A1: ABT
            t, mem, out = _measure(_run_abt, root, runs=self.runs)
            ratio = out / n if n > 0 else 0.0
            report.results.append(BenchmarkResult(
                method="ABT", h=h, n=n,
                time_ms=t, memory_kb=mem,
                output_nodes=out, compression_ratio=ratio,
            ))
            print(f"ABT {t:.2f}ms", end="  ", flush=True)

            # A2: STA
            t, mem, out = _measure(_run_sta, values, runs=self.runs)
            report.results.append(BenchmarkResult(
                method="STA", h=h, n=n,
                time_ms=t, memory_kb=mem,
                output_nodes=out, compression_ratio=out / n if n > 0 else 0.0,
            ))
            print(f"STA {t:.2f}ms", end="  ", flush=True)

            # A3: APY
            # APY operates on the leaf layer only (2^h values)
            leaf_values = values[n - (1 << h):]
            t, mem, out = _measure(_run_apy, leaf_values, runs=self.runs)
            report.results.append(BenchmarkResult(
                method="APY", h=h, n=n,
                time_ms=t, memory_kb=mem,
                output_nodes=out, compression_ratio=out / n if n > 0 else 0.0,
            ))
            print(f"APY {t:.2f}ms")

        return report


# ---------------------------------------------------------------------------
# Scalability analysis
# ---------------------------------------------------------------------------

def scalability_check(report: BenchmarkReport) -> str:
    """Verify Theta(n) scaling by checking time ratios between successive heights.

    For Theta(n) we expect time[h+2] / time[h] ≈ 4 (n quadruples per +2 height).
    Returns a text summary.
    """
    lines = ["Scalability Check (expect ~4x time per doubling of n)"]
    for method in ("ABT", "STA", "APY"):
        method_results = [r for r in report.results if r.method == method]
        method_results.sort(key=lambda r: r.h)
        lines.append(f"\n  {method}:")
        for i in range(1, len(method_results)):
            r_prev = method_results[i - 1]
            r_curr = method_results[i]
            if r_prev.time_ms > 0:
                ratio = r_curr.time_ms / r_prev.time_ms
                n_ratio = r_curr.n / r_prev.n
                lines.append(
                    f"    h={r_prev.h}→{r_curr.h}  "
                    f"n_ratio={n_ratio:.1f}x  time_ratio={ratio:.2f}x"
                )
    return "\n".join(lines)
