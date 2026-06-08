"""
tests/unit/test_extensions.py
==============================
Tests for WABT, MBT, and PBT extensions (paper Section VII).
"""

import math
import pytest

from abt.core import build_tree_from_list, node_count, height
from abt.extensions.variants import WABT, MBT, PBT
from abt.math.definitions import wabt_node_value, mbt_node_value, pbt_node_value


VALUES_H2 = [10, 6, 14, 4, 8, 12, 16]
VALUES_H1 = [20, 10, 30]


# ---------------------------------------------------------------------------
# WABT tests
# ---------------------------------------------------------------------------

class TestWABT:

    def test_uniform_weights_equals_abt(self):
        """WABT with uniform weights reduces to plain ABT."""
        from abt.core import build_abt
        T = build_tree_from_list(VALUES_H2)
        wabt = WABT.uniform(1.0)
        T_wabt = wabt.build(T)
        T_abt  = build_abt(T)
        from abt.core import bfs_values
        assert bfs_values(T_wabt) == pytest.approx(bfs_values(T_abt))

    def test_structural_theorems_hold(self):
        """WABT preserves Theorems 1 and 2 (structural guarantees)."""
        T = build_tree_from_list(VALUES_H2)
        wabt = WABT(default_weights=(2.0, 1.0))
        T_prime = wabt.build(T)
        assert node_count(T_prime) == (7 - 1) // 2
        assert height(T_prime) == 1

    def test_weighted_root_value(self):
        """Manual calculation: (2*6 + 1*14) / (2+1) = 8.667..."""
        T = build_tree_from_list(VALUES_H2)
        wabt = WABT(default_weights=(2.0, 1.0))
        T_prime = wabt.build(T)
        expected = (2.0 * 6 + 1.0 * 14) / 3.0
        assert math.isclose(T_prime.val, expected)

    def test_theorem4_containment(self):
        """Value containment holds for WABT (Theorem 4 extension)."""
        T = build_tree_from_list(VALUES_H2)
        wabt = WABT(default_weights=(3.0, 1.0))
        T_prime = wabt.build(T)
        # Root: children 6, 14 — weighted mean must be in [6, 14]
        assert 6.0 <= T_prime.val <= 14.0

    def test_equal_weights_reduces_to_abt_formula(self):
        assert math.isclose(wabt_node_value(4.0, 8.0, 1.0, 1.0), 6.0)

    def test_asymmetric_weights(self):
        # omega1=3, omega2=1: (3*4 + 1*8)/4 = 5.0
        assert math.isclose(wabt_node_value(4.0, 8.0, 3.0, 1.0), 5.0)

    def test_invalid_weight_raises(self):
        with pytest.raises(ValueError):
            wabt_node_value(4.0, 8.0, -1.0, 1.0)

    def test_none_returns_none(self):
        assert WABT().build(None) is None

    def test_height0_returns_none(self):
        T = build_tree_from_list([5.0])
        assert WABT().build(T) is None


# ---------------------------------------------------------------------------
# MBT tests
# ---------------------------------------------------------------------------

class TestMBT:

    def test_mbt_binary_equals_abt(self):
        """For binary trees, MBT == ABT (median of 2 = mean of 2)."""
        from abt.core import build_abt, bfs_values
        T = build_tree_from_list(VALUES_H2)
        T_mbt = MBT.build(T)
        T_abt = build_abt(T)
        assert bfs_values(T_mbt) == pytest.approx(bfs_values(T_abt))

    def test_structural_theorems_hold(self):
        T = build_tree_from_list(VALUES_H2)
        T_prime = MBT.build(T)
        assert node_count(T_prime) == 3
        assert height(T_prime) == 1

    def test_kary_median(self):
        """k=3: median([1,5,9]) = 5."""
        result = MBT.build_kary([1.0, 5.0, 9.0, 2.0, 6.0, 10.0], k=3)
        assert result == pytest.approx([5.0, 6.0])

    def test_median_formula(self):
        assert math.isclose(mbt_node_value(4.0, 8.0), 6.0)


# ---------------------------------------------------------------------------
# PBT tests
# ---------------------------------------------------------------------------

class TestPBT:

    def test_uniform_probs_equals_abt(self):
        """PBT with (0.5, 0.5) reduces to plain ABT."""
        from abt.core import build_abt, bfs_values
        T = build_tree_from_list(VALUES_H2)
        pbt = PBT.uniform()
        T_pbt = pbt.build(T)
        T_abt = build_abt(T)
        assert bfs_values(T_pbt) == pytest.approx(bfs_values(T_abt))

    def test_structural_theorems_hold(self):
        T = build_tree_from_list(VALUES_H2)
        pbt = PBT(default_probs=(0.7, 0.3))
        T_prime = pbt.build(T)
        assert node_count(T_prime) == 3
        assert height(T_prime) == 1

    def test_deterministic_left(self):
        """p=(1.0, 0.0): result equals left child value."""
        T = build_tree_from_list(VALUES_H2)
        pbt = PBT(default_probs=(1.0, 0.0))
        T_prime = pbt.build(T)
        # root children: 6, 14  → p1=1: result=6.0
        assert math.isclose(T_prime.val, 6.0)

    def test_deterministic_right(self):
        """p=(0.0, 1.0): result equals right child value."""
        T = build_tree_from_list(VALUES_H2)
        pbt = PBT(default_probs=(0.0, 1.0))
        T_prime = pbt.build(T)
        # root children: 6, 14  → p2=1: result=14.0
        assert math.isclose(T_prime.val, 14.0)

    def test_pbt_formula(self):
        # E[X] = 0.3*4 + 0.7*8 = 1.2 + 5.6 = 6.8
        assert math.isclose(pbt_node_value(4.0, 0.3, 8.0, 0.7), 6.8)

    def test_invalid_probs_negative(self):
        with pytest.raises(ValueError):
            pbt_node_value(4.0, -0.1, 8.0, 1.1)

    def test_invalid_probs_sum(self):
        with pytest.raises(ValueError):
            pbt_node_value(4.0, 0.3, 8.0, 0.3)  # sum = 0.6 ≠ 1

    def test_none_returns_none(self):
        assert PBT.uniform().build(None) is None

    def test_from_confidence_scores(self):
        T = build_tree_from_list(VALUES_H2)
        # confidence [2,1, 3,1, 1,3] -> probs [2/3, 1/3], [3/4, 1/4], [1/4, 3/4]
        pbt = PBT.from_confidence_scores(T, [2.0, 1.0, 3.0, 1.0, 1.0, 3.0])
        T_prime = pbt.build(T)
        assert T_prime is not None
        assert node_count(T_prime) == 3
