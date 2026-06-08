"""
abt/extensions/variants.py
==========================
Three generalisations of the ABT defined in Section VII of the paper.

  WABT – Weighted Average Binary Tree  (Definition 7, Equation 2)
  MBT  – Median Binary Tree            (Definition 8, Equation 3)
  PBT  – Probabilistic Binary Tree     (Definition 9, Equation 4)

All three share the structural guarantees of Theorems 1 and 2 because
they use the same BFS topology-copying rule (D1)(D3)(D4) and only
replace the value formula (D2).

Reference: Paper Section VII.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from typing import Callable, Dict, List, Optional, Tuple

from abt.core import (
    Node,
    build_tree_from_list,
    build_generalised_abt,
    bfs_values,
    height,
    node_count,
    validate_perfect,
)
from abt.math.definitions import wabt_node_value, mbt_node_value, pbt_node_value


# ---------------------------------------------------------------------------
# WABT — Weighted Average Binary Tree
# ---------------------------------------------------------------------------

class WABT:
    """Weighted Average Binary Tree.

    Definition 7: w'(u) = (omega1*v1 + omega2*v2) / (omega1 + omega2).

    Weights may encode sensor calibration confidence, lane priority,
    or per-channel signal-to-noise ratios (paper Section VII-A).

    Parameters
    ----------
    weights : dict mapping node id -> (omega_left, omega_right), optional
        Per-internal-node weight pairs.  When None, uniform weights
        (1.0, 1.0) are used, recovering the plain ABT.
    default_weights : tuple (omega1, omega2)
        Fallback weights for nodes not in *weights*.
    """

    def __init__(
        self,
        weights: Optional[Dict[int, Tuple[float, float]]] = None,
        default_weights: Tuple[float, float] = (1.0, 1.0),
    ) -> None:
        self.weights = weights or {}
        self.default_weights = default_weights

    def build(self, root: Optional[Node]) -> Optional[Node]:
        """Apply the WABT operator to *root*.

        Returns the WABT T' following the same BFS algorithm as ABT
        but with weighted averaging at each node.
        """
        if root is None:
            return None
        if root.left is None and root.right is None:
            return None

        validate_perfect(root)

        w1, w2 = self.weights.get(id(root), self.default_weights)
        abt_root = Node(wabt_node_value(root.left.val, root.right.val, w1, w2))

        q_orig: deque = deque([root])
        q_new:  deque = deque([abt_root])

        while q_orig:
            p = q_orig.popleft()
            u = q_new.popleft()
            if p.left.left is not None:
                wL1, wL2 = self.weights.get(id(p.left),  self.default_weights)
                wR1, wR2 = self.weights.get(id(p.right), self.default_weights)
                a_L = wabt_node_value(p.left.left.val,  p.left.right.val,  wL1, wL2)
                a_R = wabt_node_value(p.right.left.val, p.right.right.val, wR1, wR2)
                u.left  = Node(a_L)
                u.right = Node(a_R)
                q_orig.extend([p.left,  p.right])
                q_new.extend( [u.left,  u.right])

        return abt_root

    @staticmethod
    def uniform(omega: float = 1.0) -> "WABT":
        """Factory: uniform weights — reduces to plain ABT."""
        return WABT(default_weights=(omega, omega))

    @staticmethod
    def from_list(
        root: Optional[Node],
        weight_values: List[float],
    ) -> "WABT":
        """Build WABT weights from a flat list of (omega_L, omega_R) pairs.

        *weight_values* should contain 2 * (number of internal nodes) entries,
        alternating omega_L and omega_R in BFS order.
        """
        if root is None:
            return WABT()
        weights: Dict[int, Tuple[float, float]] = {}
        q: deque = deque([root])
        idx = 0
        while q and idx + 1 < len(weight_values):
            p = q.popleft()
            if p.left is not None:
                weights[id(p)] = (weight_values[idx], weight_values[idx + 1])
                idx += 2
                q.extend([p.left, p.right])
        return WABT(weights=weights)


# ---------------------------------------------------------------------------
# MBT — Median Binary Tree
# ---------------------------------------------------------------------------

class MBT:
    """Median Binary Tree.

    Definition 8: w'(u) = median(v1, v2).

    For two-element samples the median equals the mean, so MBT
    is numerically identical to ABT on binary trees but is
    architecturally distinct (and generalises differently to k-ary trees).

    Outlier robustness advantage: for k >= 3 the median is the
    ceil(k/2)-th order statistic (paper Section VII-B).
    """

    @staticmethod
    def build(root: Optional[Node]) -> Optional[Node]:
        """Apply the MBT operator to *root*."""
        return build_generalised_abt(root, mbt_node_value)

    @staticmethod
    def build_kary(values: List[float], k: int) -> List[float]:
        """Apply k-ary median aggregation to a flat list.

        Simulates one level of MBT on a k-ary perfect tree,
        grouping *values* into chunks of k and taking the median.
        """
        result = []
        for i in range(0, len(values) - k + 1, k):
            chunk = values[i : i + k]
            result.append(statistics.median(chunk))
        return result


# ---------------------------------------------------------------------------
# PBT — Probabilistic Binary Tree
# ---------------------------------------------------------------------------

class PBT:
    """Probabilistic Binary Tree.

    Definition 9: w'(u) = E[X] = p(v1)*v1 + p(v2)*v2,
    where p(v1) + p(v2) = 1.

    Each child carries a probability p(vi) in [0,1].
    Captures uncertain traffic speeds, Bayesian leaf probabilities, etc.
    (paper Section VII-C).

    Parameters
    ----------
    probs : dict mapping node id -> (p_left, p_right), optional
        Per-internal-node probability pairs summing to 1.
        Defaults to uniform (0.5, 0.5), recovering the plain ABT.
    """

    def __init__(
        self,
        probs: Optional[Dict[int, Tuple[float, float]]] = None,
        default_probs: Tuple[float, float] = (0.5, 0.5),
    ) -> None:
        self.probs = probs or {}
        self.default_probs = default_probs

    def build(self, root: Optional[Node]) -> Optional[Node]:
        """Apply the PBT operator to *root*."""
        if root is None:
            return None
        if root.left is None and root.right is None:
            return None

        validate_perfect(root)

        p1, p2 = self.probs.get(id(root), self.default_probs)
        abt_root = Node(pbt_node_value(root.left.val, p1, root.right.val, p2))

        q_orig: deque = deque([root])
        q_new:  deque = deque([abt_root])

        while q_orig:
            p = q_orig.popleft()
            u = q_new.popleft()
            if p.left.left is not None:
                pL1, pL2 = self.probs.get(id(p.left),  self.default_probs)
                pR1, pR2 = self.probs.get(id(p.right), self.default_probs)
                a_L = pbt_node_value(p.left.left.val,  pL1, p.left.right.val,  pL2)
                a_R = pbt_node_value(p.right.left.val, pR1, p.right.right.val, pR2)
                u.left  = Node(a_L)
                u.right = Node(a_R)
                q_orig.extend([p.left,  p.right])
                q_new.extend( [u.left,  u.right])

        return abt_root

    @staticmethod
    def uniform() -> "PBT":
        """Factory: uniform (0.5, 0.5) — equivalent to plain ABT."""
        return PBT(default_probs=(0.5, 0.5))

    @staticmethod
    def from_confidence_scores(
        root: Optional[Node],
        confidence: List[float],
    ) -> "PBT":
        """Build a PBT where confidence scores are normalised to probabilities.

        *confidence* should have 2 * (internal nodes) entries in BFS order,
        alternating left-child and right-child confidence values.
        """
        if root is None:
            return PBT()
        probs: Dict[int, Tuple[float, float]] = {}
        q: deque = deque([root])
        idx = 0
        while q and idx + 1 < len(confidence):
            node = q.popleft()
            if node.left is not None:
                c1, c2 = confidence[idx], confidence[idx + 1]
                total = c1 + c2
                if total <= 0:
                    p1, p2 = 0.5, 0.5
                else:
                    p1, p2 = c1 / total, c2 / total
                probs[id(node)] = (p1, p2)
                idx += 2
                q.extend([node.left, node.right])
        return PBT(probs=probs)
