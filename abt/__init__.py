"""
abt — Average Binary Tree
=========================
Python reference implementation of the Average Binary Tree (ABT) as
defined in:

  Vikas Kumar, "Average Binary Tree (ABT): A Novel Formal Tree-to-Tree
  Transformation with Pairwise Sibling Aggregation,"
  IJIRT, Volume 13 Issue 1, ISSN 2349-6002, June 2026.

Quick start
-----------
>>> from abt import ABT, Node
>>> tree = ABT.from_list([10, 6, 14, 4, 8, 12, 16])
>>> result = tree.transform()
>>> result.print()

Public API
----------
ABT             – High-level facade for the sibling-averaging operator
Node            – Single tree node with val / left / right
build_abt       – Low-level BFS operator (Algorithm 1)
build_tree_from_list – Construct a perfect binary tree from BFS-order list
WABT, MBT, PBT – Three generalisations (Section VII)
verify_all      – Run all theorem checks on a tree
"""

from abt.core import (
    Node,
    build_abt,
    build_tree_from_list,
    bfs_values,
    height,
    node_count,
    apply_f_k,
    validate_perfect,
    print_tree,
)
from abt.extensions.variants import WABT, MBT, PBT
from abt.math.theorems import verify_all, print_verification_report
from abt._facade import ABT

__all__ = [
    "ABT",
    "Node",
    "WABT",
    "MBT",
    "PBT",
    "build_abt",
    "build_tree_from_list",
    "bfs_values",
    "height",
    "node_count",
    "apply_f_k",
    "validate_perfect",
    "print_tree",
    "verify_all",
    "print_verification_report",
]

__version__ = "1.0.0"
__author__  = "Vikas Kumar"
__paper__   = (
    "Vikas Kumar, 'Average Binary Tree (ABT): A Novel Formal Tree-to-Tree "
    "Transformation with Pairwise Sibling Aggregation', "
    "IJIRT Vol 13 Issue 1, June 2026."
)
