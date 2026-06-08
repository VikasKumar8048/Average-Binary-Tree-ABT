"""
abt/core.py
===========
Core data structures for the Average Binary Tree (ABT).

Based on:
  Vikas Kumar, "Average Binary Tree (ABT): A Novel Formal Tree-to-Tree
  Transformation with Pairwise Sibling Aggregation,"
  IJIRT, Volume 13 Issue 1, June 2026.

Reference implementation follows Listings 1-4 and Algorithm 1 exactly.
"""

from __future__ import annotations

from collections import deque
from typing import Optional, List, Callable
import math


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class Node:
    """A single node in a valued binary tree.

    Attributes
    ----------
    val : float
        The real-valued payload of this node.
    left : Optional[Node]
        Left child, or None if absent.
    right : Optional[Node]
        Right child, or None if absent.
    """

    def __init__(self, val: float) -> None:
        self.val: float = float(val)
        self.left:  Optional[Node] = None
        self.right: Optional[Node] = None

    def __repr__(self) -> str:
        return f"Node({self.val})"

    def is_leaf(self) -> bool:
        """Return True iff this node has no children."""
        return self.left is None and self.right is None


# ---------------------------------------------------------------------------
# Tree construction helpers
# ---------------------------------------------------------------------------

def build_tree_from_list(values: List[float]) -> Optional[Node]:
    """Build a binary tree from a BFS-order list of values.

    This is the inverse of :func:`bfs_values`.  For a perfect binary tree
    of height h, values should have length 2^(h+1) - 1.

    Parameters
    ----------
    values : list of float
        Node values in level-order (BFS) order.

    Returns
    -------
    Node or None
        Root of the constructed tree, or None if *values* is empty.

    References: Listing 1, paper Section VI-A.
    """
    if not values:
        return None
    nodes = [Node(v) for v in values]
    for i in range(len(nodes)):
        li, ri = 2 * i + 1, 2 * i + 2
        if li < len(nodes):
            nodes[i].left = nodes[li]
        if ri < len(nodes):
            nodes[i].right = nodes[ri]
    return nodes[0]


def bfs_values(root: Optional[Node]) -> List[float]:
    """Return BFS-order list of node values.

    Parameters
    ----------
    root : Node or None

    Returns
    -------
    list of float
    """
    if root is None:
        return []
    q: deque = deque([root])
    out: List[float] = []
    while q:
        n = q.popleft()
        out.append(n.val)
        if n.left:
            q.append(n.left)
        if n.right:
            q.append(n.right)
    return out


def height(root: Optional[Node]) -> int:
    """Compute the height of a binary tree.

    Definition 2 of the paper: h(T) = max_{v in V} d_T(v).

    Returns -1 for an empty tree (no nodes).
    """
    if root is None:
        return -1
    return 1 + max(height(root.left), height(root.right))


def node_count(root: Optional[Node]) -> int:
    """Return |V|, the number of nodes in the tree."""
    if root is None:
        return 0
    return 1 + node_count(root.left) + node_count(root.right)


# ---------------------------------------------------------------------------
# Perfect-tree validator
# ---------------------------------------------------------------------------

def _leaf_depths(node: Optional[Node], d: int, out: List[int]) -> None:
    """Recursively collect leaf depths.  Raises ValueError on one-child node."""
    if node is None:
        return
    if node.left is None and node.right is None:
        out.append(d)
        return
    if (node.left is None) != (node.right is None):
        raise ValueError(
            f"Node({node.val}) has exactly one child — not a perfect binary tree."
        )
    _leaf_depths(node.left,  d + 1, out)
    _leaf_depths(node.right, d + 1, out)


def validate_perfect(root: Optional[Node]) -> None:
    """Validate that *root* is a perfect binary tree (Definition 3).

    Raises
    ------
    ValueError
        If the tree is not perfect.

    References: Listing 3, paper Section VI-A.
    """
    if root is None:
        return
    depths: List[int] = []
    _leaf_depths(root, 0, depths)
    if len(set(depths)) > 1:
        raise ValueError(
            f"Not a perfect binary tree.  Leaf depths: {sorted(set(depths))}"
        )


# ---------------------------------------------------------------------------
# Core ABT operator  — Algorithm 1
# ---------------------------------------------------------------------------

def build_abt(root: Optional[Node]) -> Optional[Node]:
    """Apply the sibling-averaging operator f : (T, w) -> T'.

    Implements Algorithm 1 (BuildABT) from the paper exactly.

    The ABT T' is constructed via a parallel BFS over the internal nodes
    of T (queue Qorig) and their images in T' (queue Qnew).  Every
    internal node p of T contributes exactly one node u = f(p) to T'
    whose value is the arithmetic mean of p's two children.

    Parameters
    ----------
    root : Node or None
        Root of a valued perfect binary tree (T, w) with h(T) >= 1.

    Returns
    -------
    Node or None
        Root of T', or None if the input is empty or a height-0 tree.

    Raises
    ------
    ValueError
        If the tree is not perfect.

    Time  : Theta(n)
    Space : Theta(n)

    References: Algorithm 1, Listings 2 & 4, paper Sections V and VI.
    """
    if root is None:
        return None
    if root.left is None and root.right is None:
        # Height-0 tree: ABT is the empty tree (paper Section V lines 3-5)
        return None

    validate_perfect(root)

    # Seed: ABT root value = mean of T root's two children   (line 7)
    abt_root = Node((root.left.val + root.right.val) / 2.0)

    q_orig: deque = deque([root])      # Qorig
    q_new:  deque = deque([abt_root])  # Qnew

    while q_orig:                               # line 9
        p = q_orig.popleft()                    # line 10
        u = q_new.popleft()                     # line 11

        if p.left.left is not None:             # line 12: p's children are internal
            a_L = (p.left.left.val  + p.left.right.val)  / 2.0   # line 13
            a_R = (p.right.left.val + p.right.right.val) / 2.0   # line 14
            u.left  = Node(a_L)                 # line 15-16
            u.right = Node(a_R)
            q_orig.extend([p.left,  p.right])   # line 17
            q_new.extend( [u.left,  u.right])   # line 18

    return abt_root


def apply_f_k(root: Optional[Node], k: int) -> Optional[Node]:
    """Apply the ABT operator k times (Corollary 3).

    f^k(T) has height h(T) - k, for 0 <= k <= h(T).
    Applying f to a height-0 tree returns the empty tree.
    """
    current = root
    for _ in range(k):
        current = build_abt(current)
        if current is None:
            break
    return current


# ---------------------------------------------------------------------------
# Generalised operator (foundation for WABT / MBT / PBT)
# ---------------------------------------------------------------------------

def build_generalised_abt(
    root: Optional[Node],
    aggregator: Callable[[float, float], float],
) -> Optional[Node]:
    """Build a generalised ABT using a custom pairwise aggregation function.

    Section VII of the paper defines WABT, MBT, and PBT as instances of
    this generalised operator where the arithmetic mean in Equation (1) is
    replaced by another aggregation function.

    Parameters
    ----------
    root : Node or None
        Root of a valued perfect binary tree.
    aggregator : callable (float, float) -> float
        Function mapping (v1, v2) -> aggregated value.

    Returns
    -------
    Node or None
    """
    if root is None:
        return None
    if root.left is None and root.right is None:
        return None

    validate_perfect(root)

    abt_root = Node(aggregator(root.left.val, root.right.val))
    q_orig: deque = deque([root])
    q_new:  deque = deque([abt_root])

    while q_orig:
        p = q_orig.popleft()
        u = q_new.popleft()
        if p.left.left is not None:
            a_L = aggregator(p.left.left.val,  p.left.right.val)
            a_R = aggregator(p.right.left.val, p.right.right.val)
            u.left  = Node(a_L)
            u.right = Node(a_R)
            q_orig.extend([p.left,  p.right])
            q_new.extend( [u.left,  u.right])

    return abt_root


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_tree(root: Optional[Node], label: str = "") -> None:
    """Print a binary tree level-by-level (BFS order).

    References: print_tree function in Listing 4.
    """
    print(f"\n--- {label} ---")
    if root is None:
        print("(empty)")
        return
    q: deque = deque([root])
    level = 0
    while q:
        row = []
        for _ in range(len(q)):
            n = q.popleft()
            row.append(n.val)
            if n.left:
                q.append(n.left)
            if n.right:
                q.append(n.right)
        print(f"  L{level}: {row}")
        level += 1


def tree_to_dict(root: Optional[Node]) -> Optional[dict]:
    """Serialise a tree to a nested dictionary (for JSON export)."""
    if root is None:
        return None
    return {
        "val":   root.val,
        "left":  tree_to_dict(root.left),
        "right": tree_to_dict(root.right),
    }


def dict_to_tree(d: Optional[dict]) -> Optional[Node]:
    """Deserialise a nested dictionary back to a Node tree."""
    if d is None:
        return None
    n = Node(d["val"])
    n.left  = dict_to_tree(d.get("left"))
    n.right = dict_to_tree(d.get("right"))
    return n
