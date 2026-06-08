"""
tests/integration/test_integration.py
======================================
End-to-end integration tests that exercise the full public API together.

Tests the ABT facade class, visualization exports, serialization,
and the paper's application scenarios.
"""

import json
import math
import os
import tempfile
import pytest

from abt import ABT, build_tree_from_list, build_abt
from abt.core import bfs_values, height, node_count, tree_to_dict, dict_to_tree
from abt.viz.visualizer import ascii_tree, to_dot, side_by_side_ascii
from abt.extensions.variants import WABT, MBT, PBT
from abt.math.theorems import verify_all


# ---------------------------------------------------------------------------
# ABT facade end-to-end
# ---------------------------------------------------------------------------

class TestABTFacade:

    def test_from_list_and_transform(self):
        tree = ABT.from_list([10, 6, 14, 4, 8, 12, 16])
        result = tree.transform()
        assert result.height == 1
        assert result.size == 3
        assert result.values() == pytest.approx([10.0, 6.0, 14.0])

    def test_verify_returns_true(self):
        tree = ABT.from_list([10, 6, 14, 4, 8, 12, 16])
        result = tree.transform()
        assert result.verify(verbose=False)

    def test_iterative_sequence_length(self):
        tree = ABT.from_list([
            100, 50, 150, 25, 75, 125, 175,
            10, 40, 60, 90, 110, 140, 160, 190
        ])
        seq = tree.iterative_sequence()
        assert len(seq) == 4  # h=3: T, f(T), f²(T), f³(T)
        assert seq[0].height == 3
        assert seq[-1].height == 0

    def test_transform_k(self):
        tree = ABT.from_list([
            100, 50, 150, 25, 75, 125, 175,
            10, 40, 60, 90, 110, 140, 160, 190
        ])
        result = tree.transform_k(2)
        assert result.height == 1
        assert result.size == 3

    def test_repr(self):
        tree = ABT.from_list([10, 6, 14, 4, 8, 12, 16])
        assert "ABT" in repr(tree)

    def test_len(self):
        tree = ABT.from_list([10, 6, 14, 4, 8, 12, 16])
        assert len(tree) == 7

    def test_empty_facade(self):
        empty = ABT()
        assert empty.is_empty
        assert empty.height == -1
        assert empty.size == 0

    def test_all_theorems_pass(self):
        for vals in [
            [10, 6, 14, 4, 8, 12, 16],
            [20, 10, 30],
            [100, 50, 150, 25, 75, 125, 175,
             10, 40, 60, 90, 110, 140, 160, 190],
        ]:
            tree = ABT.from_list(vals)
            assert tree.verify(verbose=False)


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------

class TestSerialization:

    def test_tree_to_dict_roundtrip(self):
        """tree → dict → tree should recover identical BFS values."""
        T = build_tree_from_list([10, 6, 14, 4, 8, 12, 16])
        d = tree_to_dict(T)
        T2 = dict_to_tree(d)
        assert bfs_values(T) == pytest.approx(bfs_values(T2))

    def test_dict_to_json_roundtrip(self):
        """Dict serialization must survive JSON encode/decode."""
        T = build_tree_from_list([10, 6, 14, 4, 8, 12, 16])
        T_prime = build_abt(T)
        d = tree_to_dict(T_prime)
        json_str = json.dumps(d)
        d2 = json.loads(json_str)
        T_recovered = dict_to_tree(d2)
        assert bfs_values(T_prime) == pytest.approx(bfs_values(T_recovered))

    def test_none_tree_serialization(self):
        assert tree_to_dict(None) is None
        assert dict_to_tree(None) is None


# ---------------------------------------------------------------------------
# Visualization integration
# ---------------------------------------------------------------------------

class TestVisualizationIntegration:

    def test_ascii_tree_output(self):
        T = build_tree_from_list([10, 6, 14, 4, 8, 12, 16])
        out = ascii_tree(T, "test")
        assert "L0" in out
        assert "10" in out

    def test_side_by_side_ascii(self):
        T = build_tree_from_list([10, 6, 14, 4, 8, 12, 16])
        T_prime = build_abt(T)
        out = side_by_side_ascii(T, T_prime)
        assert "original" in out.lower() or "T" in out

    def test_to_dot_output(self):
        T = build_tree_from_list([10, 6, 14, 4, 8, 12, 16])
        dot = to_dot(T)
        assert "digraph" in dot
        assert "10" in dot
        assert "->" in dot

    def test_to_dot_empty_tree(self):
        dot = to_dot(None)
        assert "digraph" in dot

    def test_save_dot_file(self):
        T = build_tree_from_list([10, 6, 14, 4, 8, 12, 16])
        with tempfile.NamedTemporaryFile(suffix=".dot", delete=False) as f:
            path = f.name
        try:
            from abt.viz.visualizer import save_dot
            save_dot(T, path)
            assert os.path.exists(path)
            with open(path) as f:
                content = f.read()
            assert "digraph" in content
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Extensions integration
# ---------------------------------------------------------------------------

class TestExtensionsIntegration:

    def test_wabt_mbt_pbt_same_structure(self):
        """All three extensions must produce trees with the same topology."""
        T = build_tree_from_list([10, 6, 14, 4, 8, 12, 16])
        T_wabt = WABT(default_weights=(2.0, 1.0)).build(T)
        T_mbt  = MBT.build(T)
        T_pbt  = PBT(default_probs=(0.6, 0.4)).build(T)

        for T_ext in (T_wabt, T_mbt, T_pbt):
            assert node_count(T_ext) == 3
            assert height(T_ext) == 1

    def test_extensions_theorem2(self):
        """Theorem 2 holds for all extensions."""
        T = build_tree_from_list([10, 6, 14, 4, 8, 12, 16])
        h_T = 2
        for T_ext in (
            WABT().build(T),
            MBT.build(T),
            PBT.uniform().build(T),
        ):
            assert height(T_ext) == h_T - 1

    def test_pipeline_chain(self):
        """Chain ABT → WABT on result → verify structural properties hold."""
        T = build_tree_from_list([
            100, 50, 150, 25, 75, 125, 175,
            10, 40, 60, 90, 110, 140, 160, 190
        ])
        # Step 1: ABT
        T1 = build_abt(T)
        assert height(T1) == 2

        # Step 2: WABT on T1 (itself a perfect tree)
        T2 = WABT(default_weights=(3.0, 1.0)).build(T1)
        assert height(T2) == 1
        assert node_count(T2) == 3

        # Step 3: MBT on T2
        T3 = MBT.build(T2)
        assert height(T3) == 0
        assert T3.is_leaf()


# ---------------------------------------------------------------------------
# Application scenario: sensor network
# ---------------------------------------------------------------------------

class TestSensorNetworkApplication:
    """Simulate Section VIII-A: Wireless Sensor Network aggregation."""

    def test_sensor_aggregation(self):
        """Leaf nodes = raw sensor readings; ABT = partial aggregates."""
        # 8 leaf sensors at h=3
        sensor_readings = [
            50.0,  # root (unused as leaf)
            40.0, 60.0,
            35.0, 45.0, 55.0, 65.0,
            30.0, 40.0, 42.0, 48.0, 52.0, 58.0, 62.0, 68.0,
        ]
        T = build_tree_from_list(sensor_readings)
        T_prime = build_abt(T)

        # ABT compresses 15 nodes to 7
        assert node_count(T_prime) == 7
        # All aggregated values are in range of original readings
        orig_range = (min(sensor_readings), max(sensor_readings))
        for v in bfs_values(T_prime):
            assert orig_range[0] <= v <= orig_range[1]

    def test_wabt_priority_sensor(self):
        """WABT with higher weight on left sensor (emergency lane)."""
        T = build_tree_from_list([50.0, 30.0, 70.0])
        wabt = WABT(default_weights=(3.0, 1.0))  # emergency lane weight=3
        T_prime = wabt.build(T)
        # (3*30 + 1*70)/4 = 160/4 = 40.0
        assert math.isclose(T_prime.val, 40.0)
        # Higher weight toward lower (emergency) reading
        assert T_prime.val < (30.0 + 70.0) / 2  # < plain mean of 50.0
