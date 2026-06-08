#!/usr/bin/env python3
"""
examples/run_benchmarks.py
==========================
Standalone benchmark runner reproducing Table III from the paper.

Compares:
  ABT  – Algorithm 1 (BFS, navigable tree output)
  STA  – Segment-Tree Aggregation (flat array)
  APY  – Array Pyramid (list of arrays)

Run:
    python examples/run_benchmarks.py
"""

import json

from abt.benchmarks.runner import ABTBenchmark, scalability_check

print("ABT Benchmark Suite — Table III from the paper")
print("=" * 55)
print("Heights: 8, 10, 12, 14  (n = 511..32767)\n")

bench = ABTBenchmark(heights=[8, 10, 12, 14], runs=5, seed=42)
report = bench.run()

print("\n" + report.summary_table())
print("\n" + scalability_check(report))

# Save outputs
with open("benchmark_table3.csv", "w") as f:
    f.write(report.to_csv())
print("\nCSV saved: benchmark_table3.csv")

results_json = [r.to_dict() for r in report.results]
with open("benchmark_table3.json", "w") as f:
    json.dump(results_json, f, indent=2)
print("JSON saved: benchmark_table3.json")
