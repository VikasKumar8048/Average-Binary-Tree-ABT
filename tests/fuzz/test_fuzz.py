"""
tests/fuzz/test_fuzz.py
=======================
Fuzz tests for the ABT core operator.

Verifies that the implementation handles unexpected or adversarial inputs
gracefully — either producing valid output or raising clear exceptions,
never crashing or silently corrupting state.
"""

import math
import random
import pytest

from abt.core import (
    Node,
    build_abt,
    build_tree_from_list,
    validate_perfect,
    height,
    node_count,
)


def random_perfect_tree(h: int, rng: random.Random) -> Node:
    n = (1 << (h + 1)) - 1
    vals = [rng.uniform(-1e9, 1e9) for _ in range(n)]
    return build_tree_from_list(vals)


class TestFuzzRandomTrees:
    """Fuzz: random perfect trees of height 1-6 must satisfy all invariants."""

    @pytest.mark.parametrize("seed", range(30))
    def test_random_trees(self, seed):
        rng = random.Random(seed)
        h = rng.randint(1, 6)
        T = random_perfect_tree(h, rng)
        T_prime = build_abt(T)

        # Theorem 1
        n = node_count(T)
        assert node_count(T_prime) == (n - 1) // 2

        # Theorem 2
        assert height(T_prime) == h - 1

        # Property 1
        assert node_count(T_prime) < n

        # Theorem 4: all ABT values in range of original values
        from abt.core import bfs_values
        orig = bfs_values(T)
        abt  = bfs_values(T_prime)
        lo, hi = min(orig), max(orig)
        for v in abt:
            assert lo - 1e-9 <= v <= hi + 1e-9, f"Out of range: {v} not in [{lo}, {hi}]"


class TestFuzzEdgeValues:
    """Fuzz: extreme numeric values must not produce NaN/Inf."""

    @pytest.mark.parametrize("val", [0.0, -0.0, 1e-308])
    def test_extreme_values_h1(self, val):
        T = build_tree_from_list([val, val, val])
        T_prime = build_abt(T)
        assert T_prime is not None
        assert math.isfinite(T_prime.val)

    @pytest.mark.parametrize("val", [1e308, -1e308])
    def test_overflow_values_h1(self, val):
        """Very large values may overflow to inf; operator still completes."""
        T = build_tree_from_list([val, val, val])
        T_prime = build_abt(T)
        # Result may be inf (float overflow) — no crash is the guarantee
        assert T_prime is not None

    def test_zero_tree(self):
        T = build_tree_from_list([0.0] * 7)
        T_prime = build_abt(T)
        from abt.core import bfs_values
        assert all(v == 0.0 for v in bfs_values(T_prime))

    def test_alternating_signs(self):
        vals = [(-1.0) ** i * float(i + 1) for i in range(7)]
        T = build_tree_from_list(vals)
        T_prime = build_abt(T)
        assert node_count(T_prime) == 3
        assert math.isfinite(T_prime.val)


class TestFuzzImperfectInputs:
    """Fuzz: imperfect trees must raise ValueError, not crash."""

    def test_single_left_child(self):
        root = Node(1.0)
        root.left = Node(0.5)
        with pytest.raises(ValueError):
            build_abt(root)

    def test_deep_right_spine(self):
        """A right-only spine is not a perfect binary tree."""
        root = Node(1.0)
        cur = root
        for i in range(5):
            cur.right = Node(float(i))
            cur = cur.right
        with pytest.raises(ValueError):
            build_abt(root)

    def test_mixed_depths(self):
        """Tree with leaves at depths 1 and 2 is not perfect."""
        root = Node(10.0)
        root.left  = Node(5.0)  # leaf at depth 1
        root.right = Node(15.0)
        root.right.left  = Node(12.0)
        root.right.right = Node(18.0)
        with pytest.raises(ValueError):
            build_abt(root)

    @pytest.mark.parametrize("seed", range(20))
    def test_random_imperfect(self, seed):
        """Randomly malformed trees must raise ValueError."""
        rng = random.Random(seed + 1000)
        h = rng.randint(2, 5)
        T = random_perfect_tree(h, rng)
        # Remove one leaf to break perfection
        from collections import deque
        q = deque([T])
        found = False
        while q and not found:
            n = q.popleft()
            if n.left and n.left.is_leaf() and n.right and n.right.is_leaf():
                n.right = None  # break the tree
                found = True
                break
            if n.left:
                q.append(n.left)
            if n.right:
                q.append(n.right)

        if found:
            with pytest.raises(ValueError):
                build_abt(T)
