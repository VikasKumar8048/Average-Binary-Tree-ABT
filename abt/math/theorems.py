"""
abt/math/theorems.py
====================
Executable verification of every theorem and property from the paper.

Theorems implemented:
  Theorem 1  – Size Contraction     |V'| = (n-1)/2
  Theorem 2  – Height Reduction     h(T') = h(T) - 1
  Corollary 3 – Iterative Contraction  h(f^k(T)) = h(T) - k
  Theorem 4  – Value Containment    min(v1,v2) <= w'(u) <= max(v1,v2)
  Theorem 5  – Operator Linearity   f(aT1 + bT2) = a·f(T1) + b·f(T2)
  Property 1 – Strict Size Decrease |V'| < |V|
  Property 2 – Isomorphism Preservation

Reference: Paper Section IV, Table II.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from abt.core import (
    Node,
    build_abt,
    build_tree_from_list,
    bfs_values,
    height,
    node_count,
    apply_f_k,
    validate_perfect,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class TheoremResult:
    name: str
    passed: bool
    expected: object
    got: object
    details: str = ""

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.name}: expected={self.expected!r} "
            f"got={self.got!r}  {self.details}"
        )


# ---------------------------------------------------------------------------
# Theorem 1 — Size Contraction
# ---------------------------------------------------------------------------

def verify_theorem1(T: Optional[Node], T_prime: Optional[Node]) -> TheoremResult:
    """Theorem 1: |V'| = (n - 1) / 2.

    Proof reference: Section IV, Theorem 1.
    """
    n = node_count(T)
    expected = (n - 1) // 2
    got = node_count(T_prime) if T_prime is not None else 0
    passed = (got == expected)
    return TheoremResult(
        name="Theorem 1 (Size Contraction)",
        passed=passed,
        expected=expected,
        got=got,
        details=f"|V|={n}, |V'|=(n-1)/2={(n-1)/2}",
    )


# ---------------------------------------------------------------------------
# Theorem 2 — Height Reduction
# ---------------------------------------------------------------------------

def verify_theorem2(T: Optional[Node], T_prime: Optional[Node]) -> TheoremResult:
    """Theorem 2: h(T') = h(T) - 1.

    Proof reference: Section IV, Theorem 2.
    """
    h_T = height(T)
    expected = h_T - 1
    got = height(T_prime) if T_prime is not None else -1
    # Special case: h(T)=0 -> T' is empty tree, height = -1
    if h_T == 0:
        expected = -1
    passed = (got == expected)
    return TheoremResult(
        name="Theorem 2 (Height Reduction)",
        passed=passed,
        expected=expected,
        got=got,
        details=f"h(T)={h_T}",
    )


# ---------------------------------------------------------------------------
# Corollary 3 — Iterative Contraction
# ---------------------------------------------------------------------------

def verify_corollary3(T: Optional[Node], k: int) -> TheoremResult:
    """Corollary 3: h(f^k(T)) = h(T) - k for 0 <= k <= h(T).

    Proof reference: Section IV, Corollary 3.
    """
    h_T = height(T)
    result = apply_f_k(T, k)
    expected_h = h_T - k
    got_h = height(result) if result is not None else -1
    # When k == h_T the result is a single node (h=0)
    if k == h_T:
        expected_h = 0
    # When k > h_T the result is empty
    if k > h_T:
        expected_h = -1
    passed = (got_h == expected_h)
    return TheoremResult(
        name=f"Corollary 3 (Iterative Contraction, k={k})",
        passed=passed,
        expected=expected_h,
        got=got_h,
        details=f"h(T)={h_T}, k={k}",
    )


# ---------------------------------------------------------------------------
# Theorem 4 — Value Containment
# ---------------------------------------------------------------------------

def _check_containment(T: Optional[Node], T_prime: Optional[Node]) -> List[Tuple]:
    """Check min(v1,v2) <= w'(u) <= max(v1,v2) at every ABT node.

    Returns list of (internal_node_val, v1, v2, abt_node_val, ok) tuples.
    """
    if T is None or T_prime is None:
        return []
    results = []
    q_orig: deque = deque([T])
    q_new:  deque = deque([T_prime])
    while q_orig and q_new:
        p = q_orig.popleft()
        u = q_new.popleft()
        v1 = p.left.val if p.left else None
        v2 = p.right.val if p.right else None
        w_u = u.val
        if v1 is not None and v2 is not None:
            lo, hi = min(v1, v2), max(v1, v2)
            ok = (lo <= w_u <= hi) or math.isclose(w_u, lo) or math.isclose(w_u, hi)
            results.append((p.val, v1, v2, w_u, ok))
            if p.left and p.left.left:
                q_orig.append(p.left)
                q_orig.append(p.right)
            if u.left:
                q_new.append(u.left)
                q_new.append(u.right)
    return results


def verify_theorem4(T: Optional[Node], T_prime: Optional[Node]) -> TheoremResult:
    """Theorem 4 (Value Containment): min(v1,v2) <= w'(u) <= max(v1,v2).

    Proof reference: Section IV, Theorem 4.
    """
    checks = _check_containment(T, T_prime)
    failures = [(p, v1, v2, w) for p, v1, v2, w, ok in checks if not ok]
    passed = len(failures) == 0
    return TheoremResult(
        name="Theorem 4 (Value Containment)",
        passed=passed,
        expected="all nodes in range",
        got=f"{len(checks) - len(failures)}/{len(checks)} nodes in range",
        details=str(failures) if failures else "all bounds satisfied",
    )


# ---------------------------------------------------------------------------
# Theorem 5 — Operator Linearity
# ---------------------------------------------------------------------------

def _add_trees(
    T1: Optional[Node], w1: float, T2: Optional[Node], w2: float
) -> Optional[Node]:
    """Return a·T1 + b·T2 as a new tree (pointwise combination)."""
    if T1 is None or T2 is None:
        return None
    vals1 = bfs_values(T1)
    vals2 = bfs_values(T2)
    if len(vals1) != len(vals2):
        raise ValueError("Trees must have identical topology for linearity check.")
    combined = [w1 * a + w2 * b for a, b in zip(vals1, vals2)]
    return build_tree_from_list(combined)


def verify_theorem5(
    T: Optional[Node],
    values1: List[float],
    values2: List[float],
    a: float,
    b: float,
) -> TheoremResult:
    """Theorem 5 (Operator Linearity): f(aT1 + bT2) = a·f(T1) + b·f(T2).

    Parameters
    ----------
    T : Node
        Topology (structure) shared by both value assignments.
    values1, values2 : list of float
        BFS-order value assignments w1 and w2 on T.
    a, b : float
        Scalar coefficients.

    Proof reference: Section IV, Theorem 5.
    """
    T1 = build_tree_from_list(values1)
    T2 = build_tree_from_list(values2)

    # LHS: f(a·T1 + b·T2)
    T_combined = _add_trees(T1, a, T2, b)
    lhs = build_abt(T_combined)
    lhs_vals = bfs_values(lhs)

    # RHS: a·f(T1) + b·f(T2)
    f_T1 = build_abt(T1)
    f_T2 = build_abt(T2)
    rhs_combined = _add_trees(f_T1, a, f_T2, b)
    rhs_vals = bfs_values(rhs_combined)

    if len(lhs_vals) != len(rhs_vals):
        return TheoremResult(
            name="Theorem 5 (Operator Linearity)",
            passed=False,
            expected="same length LHS/RHS",
            got=f"LHS len={len(lhs_vals)}, RHS len={len(rhs_vals)}",
        )

    all_close = all(
        math.isclose(l, r, rel_tol=1e-9, abs_tol=1e-12)
        for l, r in zip(lhs_vals, rhs_vals)
    )
    return TheoremResult(
        name=f"Theorem 5 (Operator Linearity, a={a}, b={b})",
        passed=all_close,
        expected="LHS == RHS (pointwise)",
        got=f"max_diff={max(abs(l-r) for l,r in zip(lhs_vals, rhs_vals)):.2e}"
            if lhs_vals else "empty",
        details=f"nodes checked: {len(lhs_vals)}",
    )


# ---------------------------------------------------------------------------
# Property 1 — Strict Size Decrease
# ---------------------------------------------------------------------------

def verify_property1(T: Optional[Node], T_prime: Optional[Node]) -> TheoremResult:
    """Property 1: |V'| < |V| for h(T) >= 1.

    Proof reference: Section IV, Property 1.
    """
    n   = node_count(T)
    n_p = node_count(T_prime) if T_prime is not None else 0
    passed = (n_p < n) if height(T) >= 1 else True
    return TheoremResult(
        name="Property 1 (Strict Size Decrease)",
        passed=passed,
        expected=f"< {n}",
        got=n_p,
        details=f"|V|={n}, |V'|={n_p}",
    )


# ---------------------------------------------------------------------------
# Property 2 — Isomorphism Preservation
# ---------------------------------------------------------------------------

def _same_topology(A: Optional[Node], B: Optional[Node]) -> bool:
    """Check structural isomorphism (ignoring values)."""
    if A is None and B is None:
        return True
    if A is None or B is None:
        return False
    return _same_topology(A.left, B.left) and _same_topology(A.right, B.right)


def verify_property2(
    T1: Optional[Node],
    T2: Optional[Node],
) -> TheoremResult:
    """Property 2: If T1 ~= T2 (topology), then f(T1) ~= f(T2).

    Proof reference: Section IV, Property 2.
    """
    if not _same_topology(T1, T2):
        return TheoremResult(
            name="Property 2 (Isomorphism Preservation)",
            passed=False,
            expected="T1 and T2 share topology",
            got="different topologies — precondition violated",
        )
    fT1 = build_abt(T1)
    fT2 = build_abt(T2)
    iso = _same_topology(fT1, fT2)
    return TheoremResult(
        name="Property 2 (Isomorphism Preservation)",
        passed=iso,
        expected="f(T1) isomorphic to f(T2)",
        got="isomorphic" if iso else "not isomorphic",
    )


# ---------------------------------------------------------------------------
# Full verification suite
# ---------------------------------------------------------------------------

def verify_all(T: Optional[Node]) -> List[TheoremResult]:
    """Run every theorem/property check on tree T.

    Returns a list of TheoremResult objects (one per theorem/property).
    """
    T_prime = build_abt(T)
    results: List[TheoremResult] = []

    results.append(verify_theorem1(T, T_prime))
    results.append(verify_theorem2(T, T_prime))

    h_T = height(T)
    for k in range(min(h_T + 1, 4)):
        results.append(verify_corollary3(T, k))

    results.append(verify_theorem4(T, T_prime))

    # Linearity with canonical a=2, b=3 and two value assignments
    bfs = bfs_values(T)
    w1 = bfs
    w2 = [v * 1.5 + 1.0 for v in bfs]
    results.append(verify_theorem5(T, w1, w2, a=2.0, b=3.0))

    results.append(verify_property1(T, T_prime))

    # Property 2: build second tree with different values, same topology
    w_alt = [v + 100.0 for v in bfs]
    T2 = build_tree_from_list(w_alt)
    results.append(verify_property2(T, T2))

    return results


def print_verification_report(T: Optional[Node], label: str = "") -> bool:
    """Print a formatted theorem verification report.  Returns True iff all pass."""
    results = verify_all(T)
    print(f"\n{'='*60}")
    print(f"Theorem Verification Report — {label}")
    print(f"{'='*60}")
    all_pass = True
    for r in results:
        print(r)
        if not r.passed:
            all_pass = False
    print(f"{'='*60}")
    print(f"Overall: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}")
    return all_pass
