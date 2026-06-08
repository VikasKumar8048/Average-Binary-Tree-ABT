"""
abt/math/proofs.py
==================
Executable proof sketches for all theorems in the paper.

Each function demonstrates the proof argument computationally on
concrete examples, confirming the mathematical reasoning.

These are *not* formal proof assistants — they are executable
illustrations of the inductive arguments used in the paper.

Reference: Paper Section IV.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from abt.core import (
    Node,
    build_abt,
    build_tree_from_list,
    bfs_values,
    height,
    node_count,
    apply_f_k,
)


# ---------------------------------------------------------------------------
# Proof of Theorem 1 — Size Contraction
# ---------------------------------------------------------------------------

def proof_theorem1(h: int) -> dict:
    """Demonstrate the proof of Theorem 1 for a perfect binary tree of height h.

    Proof structure:
      - A perfect binary tree of height h has n = 2^(h+1) - 1 nodes.
      - The leaf count is 2^h; the internal node count is n - 2^h = 2^h - 1.
      - By Definition 6(D1), |V'| = internal node count = 2^h - 1 = (n-1)/2.
    """
    n       = (1 << (h + 1)) - 1      # 2^(h+1) - 1
    leaves  = 1 << h                  # 2^h
    internal = n - leaves             # 2^h - 1

    formula = (n - 1) // 2

    assert internal == formula, f"Internal count mismatch at h={h}"

    return {
        "h":        h,
        "n":        n,
        "leaves":   leaves,
        "internal": internal,
        "formula":  formula,
        "verified": internal == formula,
        "statement": f"|V'| = {internal} = (n-1)/2 = ({n}-1)/2 = {formula}",
    }


# ---------------------------------------------------------------------------
# Proof of Theorem 2 — Height Reduction
# ---------------------------------------------------------------------------

def proof_theorem2(h: int) -> dict:
    """Demonstrate the proof of Theorem 2 for height h.

    Proof structure:
      - The deepest internal nodes of T are at depth h-1 (parents of leaves).
      - By (D1), these map to nodes at depth h-1 in T'.
      - Since they are the deepest nodes in T', h(T') = h-1.
    """
    import random
    rng = random.Random(42)
    n    = (1 << (h + 1)) - 1
    vals = [rng.uniform(0, 100) for _ in range(n)]
    T      = build_tree_from_list(vals)
    T_prime = build_abt(T)

    h_T  = height(T)
    h_Tp = height(T_prime) if T_prime is not None else -1

    return {
        "h":             h,
        "h_T":           h_T,
        "h_Tprime":      h_Tp,
        "verified":      h_Tp == h_T - 1,
        "deepest_internal_depth": h_T - 1,
        "statement": f"h(T') = {h_Tp} = h(T) - 1 = {h_T} - 1 = {h_T - 1}",
    }


# ---------------------------------------------------------------------------
# Proof of Corollary 3 — Iterative Contraction
# ---------------------------------------------------------------------------

def proof_corollary3(h: int) -> List[dict]:
    """Demonstrate Corollary 3: h(f^k(T)) = h(T) - k for 0 <= k <= h.

    Proof: Immediate from Theorem 2 by induction on k.
    """
    import random
    rng  = random.Random(7)
    n    = (1 << (h + 1)) - 1
    vals = [rng.uniform(0, 100) for _ in range(n)]
    T    = build_tree_from_list(vals)

    steps = []
    for k in range(h + 1):
        result    = apply_f_k(T, k)
        actual_h  = height(result) if result is not None else -1
        expected_h = h - k
        steps.append({
            "k":         k,
            "expected_h": expected_h,
            "actual_h":  actual_h,
            "verified":  actual_h == expected_h,
            "n_prime":   node_count(result) if result is not None else 0,
        })
    return steps


# ---------------------------------------------------------------------------
# Proof of Theorem 4 — Value Containment
# ---------------------------------------------------------------------------

def proof_theorem4_at_node(v1: float, v2: float) -> dict:
    """Demonstrate Theorem 4 at a single node.

    Proof: Assume v1 <= v2 (WLOG).
      v1 = (v1+v1)/2 <= (v1+v2)/2 <= (v2+v2)/2 = v2.

    This is the monotone mean inequality (Hardy-Littlewood-Polya [36]).
    """
    mean = (v1 + v2) / 2.0
    lo, hi = min(v1, v2), max(v1, v2)

    # Explicit inequality chain from the paper
    chain_lo = (lo + lo) / 2     # = lo
    chain_hi = (hi + hi) / 2     # = hi

    return {
        "v1":       v1,
        "v2":       v2,
        "mean":     mean,
        "lo":       lo,
        "hi":       hi,
        "chain_lo": chain_lo,
        "chain_hi": chain_hi,
        "lower_ok": chain_lo <= mean,
        "upper_ok": mean <= chain_hi,
        "verified": chain_lo <= mean <= chain_hi,
        "statement": f"{lo} = (lo+lo)/2 <= ({v1}+{v2})/2 = {mean:.4f} <= (hi+hi)/2 = {hi}",
    }


# ---------------------------------------------------------------------------
# Proof of Theorem 5 — Operator Linearity
# ---------------------------------------------------------------------------

def proof_theorem5(v1: float, v2: float, u1: float, u2: float,
                   a: float, b: float) -> dict:
    """Demonstrate Theorem 5 at a single node.

    Proof: Let w1 = (v1, v2) and w2 = (u1, u2) be sibling values.
    Combined assignment: (a*v1+b*u1, a*v2+b*u2).

    LHS: f(a*w1 + b*w2)(node) = ((a*v1+b*u1) + (a*v2+b*u2)) / 2
    RHS: a*f(T,w1) + b*f(T,w2) = a*(v1+v2)/2 + b*(u1+u2)/2

    LHS = (a*(v1+v2) + b*(u1+u2)) / 2 = a*(v1+v2)/2 + b*(u1+u2)/2 = RHS. ∎
    """
    lhs = (a * v1 + b * u1 + a * v2 + b * u2) / 2.0
    rhs = a * (v1 + v2) / 2.0 + b * (u1 + u2) / 2.0

    return {
        "v1": v1, "v2": v2, "u1": u1, "u2": u2, "a": a, "b": b,
        "lhs": lhs,
        "rhs": rhs,
        "verified": math.isclose(lhs, rhs, rel_tol=1e-12),
        "statement": (
            f"f(aT1+bT2) = ({a}*{v1}+{b}*{u1}+{a}*{v2}+{b}*{u2})/2 "
            f"= {lhs:.6f} = a*f(T1)+b*f(T2) = {rhs:.6f}"
        ),
    }


# ---------------------------------------------------------------------------
# Full proof demonstration
# ---------------------------------------------------------------------------

def run_all_proof_demonstrations() -> None:
    """Print demonstrations of all proofs from the paper."""
    print("\n" + "=" * 65)
    print("Executable Proof Demonstrations — ABT (Section IV)")
    print("=" * 65)

    print("\n─── Theorem 1: Size Contraction ───")
    for h in [1, 2, 3, 4]:
        r = proof_theorem1(h)
        status = "✓" if r["verified"] else "✗"
        print(f"  [{status}] h={h}: {r['statement']}")

    print("\n─── Theorem 2: Height Reduction ───")
    for h in [1, 2, 3, 4]:
        r = proof_theorem2(h)
        status = "✓" if r["verified"] else "✗"
        print(f"  [{status}] h={h}: {r['statement']}")

    print("\n─── Corollary 3: Iterative Contraction (h=3) ───")
    steps = proof_corollary3(3)
    for s in steps:
        status = "✓" if s["verified"] else "✗"
        print(f"  [{status}] k={s['k']}: h(f^k(T))={s['actual_h']} "
              f"(expected {s['expected_h']}), n'={s['n_prime']}")

    print("\n─── Theorem 4: Value Containment ───")
    for v1, v2 in [(4, 8), (6, 14), (25, 75), (-5, 5), (0.5, 1.5)]:
        r = proof_theorem4_at_node(float(v1), float(v2))
        status = "✓" if r["verified"] else "✗"
        print(f"  [{status}] ({v1},{v2}): {r['statement']}")

    print("\n─── Theorem 5: Operator Linearity ───")
    for args in [(4, 8, 10, 20, 2, 3), (1, 3, 2, 6, 0.5, 1.5), (0, 10, 5, 15, 1, 1)]:
        r = proof_theorem5(*[float(x) for x in args])
        status = "✓" if r["verified"] else "✗"
        print(f"  [{status}] a={args[4]}, b={args[5]}: {r['statement']}")

    print("\n" + "=" * 65)
    print("All proof demonstrations complete.")


if __name__ == "__main__":
    run_all_proof_demonstrations()
