"""
abt/math/invariants.py
======================
Runtime structural invariant checker for ABT trees.

Provides a single entry-point check_invariants() that asserts
all structural invariants hold on a given tree (T, T').
"""

from __future__ import annotations

from collections import deque
from typing import Optional, List

from abt.core import Node, validate_perfect, bfs_values, height, node_count


def check_invariants(
    T: Optional[Node],
    T_prime: Optional[Node],
    strict: bool = True,
) -> List[str]:
    """Check all structural invariants defined in the paper.

    Parameters
    ----------
    T : Node
        Original perfect binary tree.
    T_prime : Node
        ABT produced by build_abt(T).
    strict : bool
        If True, raise AssertionError on first failure.
        If False, collect and return all failure messages.

    Returns
    -------
    list of str
        Empty list if all invariants hold; error messages otherwise.
    """
    errors: List[str] = []

    def _fail(msg: str) -> None:
        if strict:
            raise AssertionError(msg)
        errors.append(msg)

    if T is None:
        if T_prime is not None:
            _fail("T is None but T' is not None.")
        return errors

    n   = node_count(T)
    h_T = height(T)

    # Invariant 1: T' size
    n_p = node_count(T_prime) if T_prime is not None else 0
    if n_p != (n - 1) // 2:
        _fail(f"Inv1 (size): |V'|={n_p} != (n-1)/2={(n-1)//2}")

    # Invariant 2: T' height
    h_p = height(T_prime) if T_prime is not None else -1
    if h_p != h_T - 1:
        _fail(f"Inv2 (height): h(T')={h_p} != h(T)-1={h_T-1}")

    # Invariant 3: T' is also a perfect binary tree
    if T_prime is not None:
        try:
            validate_perfect(T_prime)
        except ValueError as e:
            _fail(f"Inv3 (T' perfect): {e}")

    # Invariant 4: Value containment at every node
    if T is not None and T_prime is not None:
        orig_vals = bfs_values(T)
        lo, hi = min(orig_vals), max(orig_vals)
        for v in bfs_values(T_prime):
            if not (lo - 1e-9 <= v <= hi + 1e-9):
                _fail(f"Inv4 (containment): {v} not in [{lo}, {hi}]")

    # Invariant 5: strict size decrease
    if T_prime is not None and n_p >= n:
        _fail(f"Inv5 (strict decrease): |V'|={n_p} >= |V|={n}")

    return errors
