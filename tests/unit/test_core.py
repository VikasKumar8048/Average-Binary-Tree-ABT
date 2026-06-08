"""
tests/unit/test_core.py
=======================
Unit tests for the core ABT implementation.

Covers the six test categories from the paper (Section VI-B):
  1. Height-2 canonical example
  2. Height-1 minimal case
  3. Height-3 stress test
  4. Float-valued tree
  5. Single-node edge case
  6. Imperfect tree error detection

Plus additional edge-case and property tests.
"""

import math
import pytest

from abt.core import (
    Node,
    build_abt,
    build_tree_from_list,
    bfs_values,
    height,
    node_count,
    validate_perfect,
    apply_f_k,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tree(values):
    return build_tree_from_list(values)


# ---------------------------------------------------------------------------
# Test Category 1: Height-2 canonical example
# ---------------------------------------------------------------------------

class TestHeight2Canonical:
    """Section VI-B test 1: Input [10, 6, 14, 4, 8, 12, 16] (n=7)."""

    VALUES = [10, 6, 14, 4, 8, 12, 16]

    def setup_method(self):
        self.T      = make_tree(self.VALUES)
        self.T_prime = build_abt(self.T)

    def test_original_height(self):
        assert height(self.T) == 2

    def test_original_size(self):
        assert node_count(self.T) == 7

    # Theorem 1
    def test_theorem1_size_contraction(self):
        assert node_count(self.T_prime) == (7 - 1) // 2   # == 3

    # Theorem 2
    def test_theorem2_height_reduction(self):
        assert height(self.T_prime) == 1

    # Theorem 4: value containment at every node
    def test_theorem4_root(self):
        # root: children 6 and 14 → mean = 10.0
        assert math.isclose(self.T_prime.val, 10.0)
        assert 6.0 <= self.T_prime.val <= 14.0

    def test_theorem4_left_child(self):
        # left: children 4 and 8 → mean = 6.0
        assert math.isclose(self.T_prime.left.val, 6.0)
        assert 4.0 <= self.T_prime.left.val <= 8.0

    def test_theorem4_right_child(self):
        # right: children 12 and 16 → mean = 14.0
        assert math.isclose(self.T_prime.right.val, 14.0)
        assert 12.0 <= self.T_prime.right.val <= 16.0

    def test_result_levels(self):
        vals = bfs_values(self.T_prime)
        assert vals == pytest.approx([10.0, 6.0, 14.0])

    def test_result_is_perfect(self):
        # h=1 tree is perfect
        validate_perfect(self.T_prime)  # should not raise


# ---------------------------------------------------------------------------
# Test Category 2: Height-1 minimal case
# ---------------------------------------------------------------------------

class TestHeight1Minimal:
    """Section VI-B test 2: Input [20, 10, 30] — minimal non-trivial tree."""

    VALUES = [20, 10, 30]

    def setup_method(self):
        self.T       = make_tree(self.VALUES)
        self.T_prime = build_abt(self.T)

    def test_theorem1(self):
        assert node_count(self.T_prime) == (3 - 1) // 2  # == 1

    def test_theorem2(self):
        assert height(self.T_prime) == 0  # single-node tree

    def test_root_value(self):
        # mean(10, 30) = 20.0
        assert math.isclose(self.T_prime.val, 20.0)

    def test_root_is_leaf(self):
        assert self.T_prime.is_leaf()

    def test_theorem4(self):
        assert 10.0 <= self.T_prime.val <= 30.0


# ---------------------------------------------------------------------------
# Test Category 3: Height-3 stress test
# ---------------------------------------------------------------------------

class TestHeight3StressTest:
    """Section VI-B test 3: n=15, h=3."""

    VALUES = [
        100, 50, 150,
        25, 75, 125, 175,
        10, 40, 60, 90, 110, 140, 160, 190,
    ]

    def setup_method(self):
        self.T       = make_tree(self.VALUES)
        self.T_prime = build_abt(self.T)

    def test_theorem1(self):
        assert node_count(self.T_prime) == (15 - 1) // 2  # == 7

    def test_theorem2(self):
        assert height(self.T_prime) == 2

    def test_theorem4_all_nodes(self):
        """All 7 sibling-pair bounds must be verified."""
        from abt.math.theorems import _check_containment
        checks = _check_containment(self.T, self.T_prime)
        assert len(checks) > 0
        for _, v1, v2, w_u, ok in checks:
            assert ok, f"Containment failed: [{v1},{v2}] -> {w_u}"

    def test_bfs_root_value(self):
        # root children: 50, 150 → mean = 100.0
        assert math.isclose(self.T_prime.val, 100.0)


# ---------------------------------------------------------------------------
# Test Category 4: Float-valued tree
# ---------------------------------------------------------------------------

class TestFloatValues:
    """Section VI-B test 4: floating-point arithmetic."""

    VALUES = [1.0, 0.5, 1.5, 0.25, 0.75, 1.25, 1.75]

    def setup_method(self):
        self.T       = make_tree(self.VALUES)
        self.T_prime = build_abt(self.T)

    def test_theorem1(self):
        assert node_count(self.T_prime) == 3

    def test_theorem2(self):
        assert height(self.T_prime) == 1

    def test_root_value(self):
        # children 0.5, 1.5 → mean = 1.0
        assert math.isclose(self.T_prime.val, 1.0)

    def test_left_value(self):
        # children 0.25, 0.75 → mean = 0.5
        assert math.isclose(self.T_prime.left.val, 0.5)

    def test_right_value(self):
        # children 1.25, 1.75 → mean = 1.5
        assert math.isclose(self.T_prime.right.val, 1.5)

    def test_theorem4_float(self):
        vals = bfs_values(self.T)
        abt_vals = bfs_values(self.T_prime)
        assert min(vals) <= min(abt_vals)
        assert max(abt_vals) <= max(vals)


# ---------------------------------------------------------------------------
# Test Category 5: Single-node edge case
# ---------------------------------------------------------------------------

class TestSingleNode:
    """Section VI-B test 5: height-0 input returns empty ABT."""

    def test_single_node_returns_none(self):
        T = make_tree([42.0])
        T_prime = build_abt(T)
        assert T_prime is None

    def test_none_input_returns_none(self):
        assert build_abt(None) is None

    def test_height_0(self):
        T = make_tree([42.0])
        assert height(T) == 0


# ---------------------------------------------------------------------------
# Test Category 6: Imperfect tree error detection
# ---------------------------------------------------------------------------

class TestImperfectTreeDetection:
    """Section VI-B test 6: ValueError on imperfect input."""

    def test_one_child_raises(self):
        """Root with only a left child is not perfect."""
        root = Node(10.0)
        root.left = Node(5.0)
        with pytest.raises(ValueError, match="one child"):
            build_abt(root)

    def test_uneven_leaf_depths_raises(self):
        """Leaf at depth 1 and leaf at depth 2 — not perfect."""
        root = Node(10.0)
        root.left  = Node(5.0)
        root.right = Node(15.0)
        root.right.left  = Node(12.0)
        root.right.right = Node(18.0)
        with pytest.raises(ValueError):
            build_abt(root)

    def test_validate_perfect_raises(self):
        root = Node(1.0)
        root.left = Node(0.5)
        with pytest.raises(ValueError):
            validate_perfect(root)


# ---------------------------------------------------------------------------
# Additional property-based checks
# ---------------------------------------------------------------------------

class TestProperties:

    @pytest.mark.parametrize("h", [1, 2, 3, 4])
    def test_theorem1_all_heights(self, h):
        """Theorem 1 holds for all perfect binary trees h=1..4."""
        import random
        rng = random.Random(0)
        n = (1 << (h + 1)) - 1
        vals = [rng.uniform(0, 100) for _ in range(n)]
        T = make_tree(vals)
        T_prime = build_abt(T)
        assert node_count(T_prime) == (n - 1) // 2

    @pytest.mark.parametrize("h", [1, 2, 3, 4])
    def test_theorem2_all_heights(self, h):
        """Theorem 2 holds for all perfect binary trees h=1..4."""
        import random
        rng = random.Random(1)
        n = (1 << (h + 1)) - 1
        vals = [rng.uniform(0, 100) for _ in range(n)]
        T = make_tree(vals)
        T_prime = build_abt(T)
        assert height(T_prime) == h - 1

    def test_property1_strict_decrease(self):
        """Property 1: |V'| < |V|."""
        T = make_tree([10, 6, 14, 4, 8, 12, 16])
        T_prime = build_abt(T)
        assert node_count(T_prime) < node_count(T)

    def test_corollary3_iterative(self):
        """Corollary 3: h(f^k(T)) = h(T) - k."""
        T = make_tree([100, 50, 150, 25, 75, 125, 175,
                       10, 40, 60, 90, 110, 140, 160, 190])
        h0 = height(T)  # == 3
        for k in range(h0 + 1):
            result = apply_f_k(T, k)
            if k < h0:
                assert height(result) == h0 - k
            else:
                # k == h0: single-node or empty
                assert result is not None and result.is_leaf()

    def test_negative_values(self):
        """ABT handles negative values correctly."""
        T = make_tree([-4, -8, 0, -10, -6, -2, 2])
        T_prime = build_abt(T)
        assert math.isclose(T_prime.val, (-8 + 0) / 2)
        assert math.isclose(T_prime.left.val,  (-10 + -6) / 2)
        assert math.isclose(T_prime.right.val, (-2 + 2) / 2)

    def test_large_values(self):
        """ABT handles large floating-point values."""
        T = make_tree([1e12, 5e11, 1.5e12, 2.5e11, 7.5e11, 1.25e12, 1.75e12])
        T_prime = build_abt(T)
        assert math.isfinite(T_prime.val)

    def test_identical_values(self):
        """When all values are equal, ABT values equal the same constant."""
        T = make_tree([5.0] * 7)
        T_prime = build_abt(T)
        for v in bfs_values(T_prime):
            assert math.isclose(v, 5.0)
