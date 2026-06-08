"""
tests/property/test_properties.py
==================================
Property-based tests using Hypothesis.

These tests verify that the formal theorems hold for *randomly generated*
perfect binary trees across all reachable heights and value distributions,
providing much broader coverage than hand-written examples.

Reference: Paper Section IV (theorems hold for all perfect binary trees).
"""

import math
import sys

import pytest

try:
    from hypothesis import given, settings, assume, HealthCheck
    from hypothesis import strategies as st
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False

from abt.core import (
    build_abt,
    build_tree_from_list,
    bfs_values,
    height,
    node_count,
    apply_f_k,
)
from abt.math.theorems import (
    verify_theorem1,
    verify_theorem2,
    verify_theorem4,
    verify_property1,
    verify_property2,
)

pytestmark = pytest.mark.skipif(
    not HAS_HYPOTHESIS,
    reason="hypothesis not installed — run: pip install hypothesis"
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def perfect_tree_strategy(max_height: int = 4):
    """Generate a random perfect binary tree of height 1..max_height."""

    @st.composite
    def _build(draw):
        h = draw(st.integers(min_value=1, max_value=max_height))
        n = (1 << (h + 1)) - 1
        values = draw(
            st.lists(
                st.floats(min_value=-1e6, max_value=1e6,
                          allow_nan=False, allow_infinity=False),
                min_size=n, max_size=n,
            )
        )
        return build_tree_from_list(values)

    return _build()


# ---------------------------------------------------------------------------
# Theorem 1 — Size Contraction
# ---------------------------------------------------------------------------

@given(T=perfect_tree_strategy(max_height=5))
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_theorem1_property(T):
    """Theorem 1: |V'| = (n-1)/2 for every perfect binary tree."""
    T_prime = build_abt(T)
    result = verify_theorem1(T, T_prime)
    assert result.passed, str(result)


# ---------------------------------------------------------------------------
# Theorem 2 — Height Reduction
# ---------------------------------------------------------------------------

@given(T=perfect_tree_strategy(max_height=5))
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_theorem2_property(T):
    """Theorem 2: h(T') = h(T)-1 for every perfect binary tree."""
    T_prime = build_abt(T)
    result = verify_theorem2(T, T_prime)
    assert result.passed, str(result)


# ---------------------------------------------------------------------------
# Theorem 4 — Value Containment
# ---------------------------------------------------------------------------

@given(T=perfect_tree_strategy(max_height=5))
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_theorem4_property(T):
    """Theorem 4: min(v1,v2) <= w'(u) <= max(v1,v2) at every ABT node."""
    T_prime = build_abt(T)
    result = verify_theorem4(T, T_prime)
    assert result.passed, str(result)


# ---------------------------------------------------------------------------
# Property 1 — Strict Size Decrease
# ---------------------------------------------------------------------------

@given(T=perfect_tree_strategy(max_height=5))
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_property1_property(T):
    """Property 1: |V'| < |V| for all h >= 1."""
    T_prime = build_abt(T)
    result = verify_property1(T, T_prime)
    assert result.passed, str(result)


# ---------------------------------------------------------------------------
# Property 2 — Isomorphism Preservation
# ---------------------------------------------------------------------------

@given(
    vals1=st.lists(
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        min_size=7, max_size=7,
    ),
    vals2=st.lists(
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        min_size=7, max_size=7,
    ),
)
@settings(max_examples=200)
def test_property2_property(vals1, vals2):
    """Property 2: f(T1) ~= f(T2) whenever T1 ~= T2 (same h=2 topology)."""
    T1 = build_tree_from_list(vals1)
    T2 = build_tree_from_list(vals2)
    result = verify_property2(T1, T2)
    assert result.passed, str(result)


# ---------------------------------------------------------------------------
# Corollary 3 — Iterative Contraction
# ---------------------------------------------------------------------------

@given(
    h=st.integers(min_value=1, max_value=4),
    seed=st.integers(min_value=0, max_value=999),
)
@settings(max_examples=100)
def test_corollary3_property(h, seed):
    """Corollary 3: h(f^k(T)) = h(T) - k for all valid k."""
    import random
    rng = random.Random(seed)
    n = (1 << (h + 1)) - 1
    vals = [rng.uniform(-100, 100) for _ in range(n)]
    T = build_tree_from_list(vals)

    for k in range(h + 1):
        result = apply_f_k(T, k)
        if k < h:
            assert height(result) == h - k, f"k={k}: expected h={h-k}, got {height(result)}"
        else:
            # k == h: single node remains
            assert result is not None and result.is_leaf()


# ---------------------------------------------------------------------------
# Theorem 5 — Operator Linearity
# ---------------------------------------------------------------------------

@given(
    vals=st.lists(
        st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False),
        min_size=7, max_size=7,
    ),
    a=st.floats(min_value=-5, max_value=5, allow_nan=False, allow_infinity=False),
    b=st.floats(min_value=-5, max_value=5, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200)
def test_theorem5_linearity_property(vals, a, b):
    """Theorem 5: f(a*T1 + b*T2) == a*f(T1) + b*f(T2) pointwise."""
    from abt.math.theorems import verify_theorem5
    w2 = [v * 1.1 + 0.5 for v in vals]
    result = verify_theorem5(
        build_tree_from_list(vals), vals, w2, a=a, b=b
    )
    assert result.passed, str(result)
