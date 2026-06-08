"""
abt/math/operators.py
=====================
Standard aggregation operators for use with build_generalised_abt().

Implements the four operators from the paper (Table IV):
  arithmetic_mean  — plain ABT (Eq. 1)
  weighted_mean    — WABT (Eq. 2)
  median           — MBT  (Eq. 3)
  probabilistic    — PBT  (Eq. 4)

Plus additional operators for research extensions:
  geometric_mean
  harmonic_mean
  max_operator
  min_operator
"""

from __future__ import annotations

import math
import statistics
from typing import Callable


# ---------------------------------------------------------------------------
# Paper operators
# ---------------------------------------------------------------------------

def arithmetic_mean(v1: float, v2: float) -> float:
    """Equation (1): w'(u) = (v1 + v2) / 2."""
    return (v1 + v2) / 2.0


def weighted_mean(omega1: float, omega2: float) -> Callable[[float, float], float]:
    """Equation (2) factory: w'(u) = (omega1*v1 + omega2*v2) / (omega1+omega2)."""
    if omega1 <= 0 or omega2 <= 0:
        raise ValueError("Weights must be positive.")
    total = omega1 + omega2
    return lambda v1, v2: (omega1 * v1 + omega2 * v2) / total


def median_operator(v1: float, v2: float) -> float:
    """Equation (3): w'(u) = median(v1, v2) = (v1+v2)/2 for two values."""
    return statistics.median([v1, v2])


def probabilistic(p1: float, p2: float) -> Callable[[float, float], float]:
    """Equation (4) factory: w'(u) = p1*v1 + p2*v2, where p1+p2=1."""
    if p1 < 0 or p2 < 0:
        raise ValueError("Probabilities must be non-negative.")
    if not math.isclose(p1 + p2, 1.0, abs_tol=1e-9):
        raise ValueError(f"Probabilities must sum to 1, got {p1+p2}.")
    return lambda v1, v2: p1 * v1 + p2 * v2


# ---------------------------------------------------------------------------
# Extended operators
# ---------------------------------------------------------------------------

def geometric_mean(v1: float, v2: float) -> float:
    """Geometric mean: sqrt(|v1 * v2|), with sign of product."""
    product = v1 * v2
    sign = 1.0 if product >= 0 else -1.0
    return sign * math.sqrt(abs(product))


def harmonic_mean(v1: float, v2: float) -> float:
    """Harmonic mean: 2*v1*v2 / (v1+v2). Requires both values non-zero."""
    if v1 + v2 == 0:
        return 0.0
    return 2.0 * v1 * v2 / (v1 + v2)


def max_operator(v1: float, v2: float) -> float:
    """Max-pooling operator: max(v1, v2)."""
    return max(v1, v2)


def min_operator(v1: float, v2: float) -> float:
    """Min-pooling operator: min(v1, v2)."""
    return min(v1, v2)
