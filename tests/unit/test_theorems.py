"""
tests/unit/test_theorems.py
===========================
Tests for all formal theorems and properties.

Verifies executable theorem checking against multiple tree instances.
"""

import math
import pytest

from abt.core import build_tree_from_list, build_abt, height, node_count
from abt.math.theorems import (
    verify_theorem1,
    verify_theorem2,
    verify_corollary3,
    verify_theorem4,
    verify_theorem5,
    verify_property1,
    verify_property2,
    verify_all,
    print_verification_report,
)

CANONICAL_H2  = [10, 6, 14, 4, 8, 12, 16]
CANONICAL_H1  = [20, 10, 30]
CANONICAL_H3  = [100, 50, 150, 25, 75, 125, 175,
                  10, 40, 60, 90, 110, 140, 160, 190]
CANONICAL_FLOAT = [1.0, 0.5, 1.5, 0.25, 0.75, 1.25, 1.75]


@pytest.mark.parametrize("values,label", [
    (CANONICAL_H2,    "h=2"),
    (CANONICAL_H1,    "h=1"),
    (CANONICAL_H3,    "h=3"),
    (CANONICAL_FLOAT, "float"),
])
class TestAllTheorems:

    def _trees(self, values):
        T = build_tree_from_list(values)
        T_prime = build_abt(T)
        return T, T_prime

    def test_theorem1(self, values, label):
        T, Tp = self._trees(values)
        result = verify_theorem1(T, Tp)
        assert result.passed, f"{label}: {result}"

    def test_theorem2(self, values, label):
        T, Tp = self._trees(values)
        result = verify_theorem2(T, Tp)
        assert result.passed, f"{label}: {result}"

    def test_theorem4(self, values, label):
        T, Tp = self._trees(values)
        result = verify_theorem4(T, Tp)
        assert result.passed, f"{label}: {result}"

    def test_property1(self, values, label):
        T, Tp = self._trees(values)
        result = verify_property1(T, Tp)
        assert result.passed, f"{label}: {result}"

    def test_property2(self, values, label):
        T, _ = self._trees(values)
        T2 = build_tree_from_list([v + 100 for v in values])
        result = verify_property2(T, T2)
        assert result.passed, f"{label}: {result}"


@pytest.mark.parametrize("values,k", [
    (CANONICAL_H2, 0),
    (CANONICAL_H2, 1),
    (CANONICAL_H2, 2),
    (CANONICAL_H3, 1),
    (CANONICAL_H3, 2),
    (CANONICAL_H3, 3),
])
def test_corollary3(values, k):
    T = build_tree_from_list(values)
    result = verify_corollary3(T, k)
    assert result.passed, f"k={k}: {result}"


@pytest.mark.parametrize("values,a,b", [
    (CANONICAL_H2, 1.0, 1.0),
    (CANONICAL_H2, 2.0, 3.0),
    (CANONICAL_H2, 0.5, -0.5),
    (CANONICAL_H3, 1.0, 1.0),
    (CANONICAL_H1, 3.0, 1.0),
])
def test_theorem5_linearity(values, a, b):
    w2 = [v * 2 + 1 for v in values]
    result = verify_theorem5(
        build_tree_from_list(values), values, w2, a=a, b=b
    )
    assert result.passed, f"a={a},b={b}: {result}"


def test_verify_all_canonical():
    T = build_tree_from_list(CANONICAL_H2)
    results = verify_all(T)
    failed = [r for r in results if not r.passed]
    assert len(failed) == 0, f"Failed theorems: {failed}"


def test_print_verification_report(capsys):
    T = build_tree_from_list(CANONICAL_H2)
    all_pass = print_verification_report(T, label="h=2 canonical")
    assert all_pass
    captured = capsys.readouterr()
    assert "PASS" in captured.out
    assert "FAIL" not in captured.out
