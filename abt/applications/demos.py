"""
abt/applications/demos.py
=========================
Demonstrations of ABT in real-world application domains.

Implements four application scenarios from Section VIII of the paper:

  A. Wireless Sensor Networks   (Section VIII-A)
  B. Multi-Resolution Image Analysis  (Section VIII-B)
  C. Traffic Flow Modelling     (Section VIII-C)
  D. MapReduce Distributed Aggregation  (Section VIII-D)
  E. Decision-Tree Smoothing    (Section VIII-E)

Each demo includes:
  - Dataset generation
  - ABT/extension construction
  - Metrics / analysis
  - Summary report

Reference: Paper Section VIII.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from abt.core import (
    Node,
    build_abt,
    build_tree_from_list,
    bfs_values,
    height,
    node_count,
    apply_f_k,
)
from abt.extensions.variants import WABT, MBT, PBT


# ---------------------------------------------------------------------------
# A. Wireless Sensor Networks  (Section VIII-A)
# ---------------------------------------------------------------------------

@dataclass
class SensorReading:
    sensor_id: int
    value: float          # e.g. temperature °C
    confidence: float     # calibration confidence in [0, 1]


def demo_wireless_sensor_network(
    height: int = 3,
    seed: int = 42,
) -> Dict:
    """Simulate a binary-tree sensor topology and apply ABT aggregation.

    Leaf nodes carry raw physical readings (temperature).
    Internal relay nodes aggregate children before forwarding to base station.

    Reproduces the scenario in Section VIII-A.

    Parameters
    ----------
    height : int
        Height of the sensor tree (number of relay levels).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with fields: original, abt, reduction_pct, report
    """
    rng = random.Random(seed)
    n_leaves = 1 << height   # 2^height leaf sensors

    # Generate leaf readings: temperature in [15, 35] °C
    leaf_readings = [rng.uniform(15.0, 35.0) for _ in range(n_leaves)]

    # Build full perfect binary tree (BFS order)
    # Internal nodes initialised to 0.0 (overwritten by ABT)
    n_total = (1 << (height + 1)) - 1
    values = [0.0] * (n_total - n_leaves) + leaf_readings
    T = build_tree_from_list(values)

    # Build WABT: confidence scores as weights
    confidences = [rng.uniform(0.5, 1.0) for _ in range(n_total)]
    wabt = WABT.from_list(T, confidences)
    T_wabt = wabt.build(T)

    # Also build plain ABT for comparison
    T_abt = build_abt(T)

    orig_nodes = node_count(T)
    abt_nodes  = node_count(T_abt)
    reduction  = (orig_nodes - abt_nodes) / orig_nodes * 100.0

    report = (
        f"=== Wireless Sensor Network Demo (h={height}) ===\n"
        f"  Leaf sensors  : {n_leaves}\n"
        f"  Total nodes   : {orig_nodes}\n"
        f"  ABT nodes     : {abt_nodes}\n"
        f"  Data reduction: {reduction:.1f}% (≈50%, consistent with Theorem 1)\n"
        f"  ABT root value: {T_abt.val:.2f}°C (mean of sub-tree)\n"
        f"  WABT root val : {T_wabt.val:.2f}°C (confidence-weighted)\n"
        f"  Value range   : [{min(leaf_readings):.2f}, {max(leaf_readings):.2f}]°C\n"
        f"  Thm 4 check   : {min(leaf_readings):.2f} <= {T_abt.val:.2f} <= {max(leaf_readings):.2f} "
        f"{'✓' if min(leaf_readings) <= T_abt.val <= max(leaf_readings) else '✗'}\n"
    )
    return {
        "original": T,
        "abt": T_abt,
        "wabt": T_wabt,
        "leaf_readings": leaf_readings,
        "reduction_pct": reduction,
        "report": report,
    }


# ---------------------------------------------------------------------------
# B. Multi-Resolution Image Analysis  (Section VIII-B)
# ---------------------------------------------------------------------------

def demo_image_pyramid(signal_length: int = 16, seed: int = 0) -> Dict:
    """Demonstrate ABT as a 1-D image/signal pyramid (Gaussian pyramid analogue).

    On a 1-D signal sampled at 2^h points arranged in a complete binary tree,
    one level of pyramid reduction is exactly one application of the ABT operator.

    Reproduces the observation in Section VIII-B.

    Parameters
    ----------
    signal_length : int
        Number of leaf signal samples (must be a power of 2).
    seed : int

    Returns
    -------
    dict with fields: levels, report
    """
    if signal_length & (signal_length - 1):
        raise ValueError("signal_length must be a power of 2.")

    rng = random.Random(seed)
    h = int(math.log2(signal_length))

    # Create a signal with a few distinct frequencies
    samples = [
        math.sin(2 * math.pi * i / signal_length) * 100 + rng.uniform(-5, 5)
        for i in range(signal_length)
    ]

    # Build perfect binary tree (leaves = signal samples)
    n_total = (1 << (h + 1)) - 1
    values = [0.0] * (n_total - signal_length) + samples
    T = build_tree_from_list(values)

    # Build full iterative contraction sequence (multi-resolution pyramid)
    from abt import ABT
    tree = ABT.from_node(T)
    sequence = tree.iterative_sequence()

    # Each level is one pyramid level (Section VIII-B)
    levels = [(i, t.height, t.size, t.values()) for i, t in enumerate(sequence)]

    report = [f"=== 1-D Signal Pyramid Demo (signal_length={signal_length}) ==="]
    for lvl, h_t, n_t, vals in levels:
        rng_val = max(vals) - min(vals) if vals else 0
        report.append(
            f"  Level {lvl}: h={h_t}, n={n_t:4d}, "
            f"range={rng_val:.2f}, "
            f"mean={sum(vals)/len(vals):.2f}"
        )
    report.append(
        f"\n  Corollary 3 verification: "
        f"{'✓' if len(sequence) == h + 1 else '✗'} "
        f"(sequence length = h+1 = {h+1})"
    )

    return {"levels": levels, "original_signal": samples, "report": "\n".join(report)}


# ---------------------------------------------------------------------------
# C. Traffic Flow Modelling  (Section VIII-C)
# ---------------------------------------------------------------------------

def demo_traffic_flow(height: int = 3, seed: int = 7) -> Dict:
    """Model hierarchical road network as a binary tree.

    Leaf nodes carry individual lane vehicle-count measurements.
    ABT yields a compressed tree of mean throughput at coarser spatial scales.
    WABT assigns higher weight to emergency lanes.

    Reproduces Section VIII-C.

    Returns
    -------
    dict
    """
    rng = random.Random(seed)
    n_leaves = 1 << height

    # Lane counts: vehicles per minute in [10, 120]
    lane_counts = [rng.randint(10, 120) for _ in range(n_leaves)]
    # Emergency lanes: every 4th lane gets higher weight
    lane_weights = [
        3.0 if i % 4 == 0 else 1.0
        for i in range(n_leaves)
    ]

    n_total = (1 << (height + 1)) - 1
    values = [0.0] * (n_total - n_leaves) + [float(c) for c in lane_counts]
    T = build_tree_from_list(values)

    # Flat weight list for WABT (broadcast to all levels)
    all_weights = lane_weights * (n_total // n_leaves + 1)
    all_weights = all_weights[:n_total]

    wabt = WABT.from_list(T, all_weights)
    T_wabt = wabt.build(T)
    T_abt  = build_abt(T)

    report = (
        f"=== Traffic Flow Modelling Demo (h={height}) ===\n"
        f"  Total lanes     : {n_leaves}\n"
        f"  Emergency lanes : {sum(1 for w in lane_weights if w > 1.0)} "
        f"(weight=3×)\n"
        f"  ABT root (mean throughput)  : {T_abt.val:.1f} vehicles/min\n"
        f"  WABT root (priority-weighted): {T_wabt.val:.1f} vehicles/min\n"
        f"  Lane counts range: [{min(lane_counts)}, {max(lane_counts)}]\n"
    )
    return {
        "original": T,
        "abt": T_abt,
        "wabt": T_wabt,
        "lane_counts": lane_counts,
        "lane_weights": lane_weights,
        "report": report,
    }


# ---------------------------------------------------------------------------
# D. MapReduce Distributed Aggregation  (Section VIII-D)
# ---------------------------------------------------------------------------

def demo_mapreduce(height: int = 3, seed: int = 3) -> Dict:
    """Model MapReduce reduce phase as a binary tree of partial averages.

    The intermediate tree of partial averages produced at each level is
    precisely the ABT of the input-value tree.

    Theorem 1 quantifies the data-volume reduction at each level (≈50%).
    Theorem 5 justifies commutativity of distributed averaging with
    subsequent linear post-processing.

    Reproduces Section VIII-D.

    Returns
    -------
    dict
    """
    rng = random.Random(seed)
    n_leaves = 1 << height

    # Mapper outputs: key-value pairs (simulated as floats)
    mapper_outputs = [rng.uniform(0.0, 1000.0) for _ in range(n_leaves)]

    n_total = (1 << (height + 1)) - 1
    values = [0.0] * (n_total - n_leaves) + mapper_outputs
    T = build_tree_from_list(values)

    # Each application of f = one reduce level
    from abt import ABT
    tree = ABT.from_node(T)
    seq = tree.iterative_sequence()

    # Verify Theorem 1 at each level
    thm1_checks = []
    for i in range(1, len(seq)):
        n_prev = seq[i - 1].size
        n_curr = seq[i].size
        expected = (n_prev - 1) // 2
        thm1_checks.append((i, n_prev, n_curr, expected, n_curr == expected))

    all_ok = all(ok for _, _, _, _, ok in thm1_checks)
    report_lines = [f"=== MapReduce Distributed Aggregation Demo (h={height}) ==="]
    for level, n_prev, n_curr, exp, ok_flag in thm1_checks:
        report_lines.append(
            f"  Reduce level {level}: nodes {n_prev} → {n_curr} "
            f"(expected {exp}) {'✓' if ok_flag else '✗'}"
        )
    reduction = (1.0 - seq[-1].size / seq[0].size) * 100.0
    report_lines.append(
        f"  Total data reduction: {reduction:.1f}% over {height} reduce levels\n"
        f"  Theorem 1 at all levels: {'✓' if all_ok else '✗'}"
    )

    return {
        "sequence": seq,
        "mapper_outputs": mapper_outputs,
        "thm1_all_pass": all_ok,
        "report": "\n".join(report_lines),
    }


# ---------------------------------------------------------------------------
# E. Decision-Tree Smoothing  (Section VIII-E)
# ---------------------------------------------------------------------------

def demo_decision_tree_smoothing(height: int = 3, seed: int = 9) -> Dict:
    """Smooth a decision tree's leaf values by applying the ABT operator.

    Replacing each sibling leaf pair with their average produces a
    step-function approximation with half the number of distinct decision
    regions, reducing over-fitting (Section VIII-E).

    The PBT variant handles probabilistic leaf outputs (class probabilities).

    Returns
    -------
    dict
    """
    rng = random.Random(seed)
    n_leaves = 1 << height

    # Leaf values: class-probability estimates in [0, 1]
    leaf_probs = [rng.uniform(0.0, 1.0) for _ in range(n_leaves)]

    n_total = (1 << (height + 1)) - 1
    values = [0.0] * (n_total - n_leaves) + leaf_probs
    T = build_tree_from_list(values)

    T_abt = build_abt(T)  # smooth leaf pairs

    # PBT variant: use leaf probabilities as weighting
    pbt_conf = leaf_probs + [0.5] * (n_total - n_leaves)
    pbt = PBT.from_confidence_scores(T, pbt_conf[:n_total])
    T_pbt = pbt.build(T)

    # Distinct decision regions = distinct leaf values (approx)
    orig_regions = len(set(round(v, 4) for v in leaf_probs))
    abt_regions  = len(set(round(v, 4) for v in bfs_values(T_abt)))

    report = (
        f"=== Decision-Tree Smoothing Demo (h={height}) ===\n"
        f"  Original leaves  : {n_leaves}  distinct regions: ~{orig_regions}\n"
        f"  After ABT smooth : {node_count(T_abt)} nodes  "
        f"distinct regions: ~{abt_regions}\n"
        f"  Region reduction : "
        f"{(orig_regions - abt_regions) / orig_regions * 100:.1f}%\n"
        f"  ABT root (mean prob): {T_abt.val:.4f}\n"
        f"  PBT root (weighted) : {T_pbt.val:.4f}\n"
    )
    return {
        "original": T,
        "abt": T_abt,
        "pbt": T_pbt,
        "leaf_probs": leaf_probs,
        "orig_regions": orig_regions,
        "abt_regions": abt_regions,
        "report": report,
    }


# ---------------------------------------------------------------------------
# Run all demos
# ---------------------------------------------------------------------------

def run_all_demos() -> None:
    """Run all five application demos and print their reports."""
    print("\n" + "=" * 60)
    print("ABT Real-World Application Demonstrations (Section VIII)")
    print("=" * 60)

    demos = [
        ("A. Wireless Sensor Networks",   lambda: demo_wireless_sensor_network()),
        ("B. 1-D Signal Pyramid",          lambda: demo_image_pyramid()),
        ("C. Traffic Flow Modelling",      lambda: demo_traffic_flow()),
        ("D. MapReduce Aggregation",       lambda: demo_mapreduce()),
        ("E. Decision-Tree Smoothing",     lambda: demo_decision_tree_smoothing()),
    ]

    for title, fn in demos:
        print(f"\n{title}")
        result = fn()
        print(result["report"])
