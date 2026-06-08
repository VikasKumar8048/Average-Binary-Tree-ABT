"""
abt/_facade.py
==============
High-level ABT facade class.

Wraps the low-level operator functions into a convenient object-oriented
API for everyday use and teaching purposes.
"""

from __future__ import annotations

from typing import List, Optional

from abt.core import (
    Node,
    build_abt,
    build_tree_from_list,
    bfs_values,
    height as _height,
    node_count as _node_count,
    apply_f_k,
    print_tree,
)
from abt.math.theorems import verify_all, TheoremResult


class ABT:
    """High-level facade for the Average Binary Tree operator.

    Parameters
    ----------
    root : Node or None
        Root of the underlying valued perfect binary tree.

    Examples
    --------
    >>> tree = ABT.from_list([10, 6, 14, 4, 8, 12, 16])
    >>> result = tree.transform()
    >>> result.print()
    >>> all_pass = result.verify()
    """

    def __init__(self, root: Optional[Node] = None) -> None:
        self._root = root

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_list(cls, values: List[float]) -> "ABT":
        """Build an ABT wrapper from a BFS-order list of node values.

        Parameters
        ----------
        values : list of float
            Node values in level-order for a perfect binary tree.
            Length must be 2^(h+1) - 1 for some h >= 0.

        Returns
        -------
        ABT
        """
        root = build_tree_from_list(values)
        return cls(root)

    @classmethod
    def from_node(cls, root: Optional[Node]) -> "ABT":
        """Wrap an existing Node tree."""
        return cls(root)

    # ------------------------------------------------------------------
    # Core transformation — f : T -> T'
    # ------------------------------------------------------------------

    def transform(self) -> "ABT":
        """Apply the sibling-averaging operator once.

        Returns
        -------
        ABT
            A new ABT wrapping T' = f(T, w).
        """
        new_root = build_abt(self._root)
        return ABT(new_root)

    def transform_k(self, k: int) -> "ABT":
        """Apply the operator k times (Corollary 3).

        Returns
        -------
        ABT
            A new ABT wrapping f^k(T).
        """
        new_root = apply_f_k(self._root, k)
        return ABT(new_root)

    def iterative_sequence(self) -> List["ABT"]:
        """Return the full iterative contraction sequence.

        Returns [T, f(T), f^2(T), ..., f^h(T)] as a list of ABT objects.
        Corollary 3: h(f^k(T)) = h(T) - k.
        """
        sequence = [self]
        current = self
        while True:
            next_abt = current.transform()
            if next_abt.root is None:
                break
            sequence.append(next_abt)
            current = next_abt
        return sequence

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def root(self) -> Optional[Node]:
        """Root node of the underlying tree."""
        return self._root

    @property
    def height(self) -> int:
        """h(T): height of the tree (-1 for empty)."""
        return _height(self._root)

    @property
    def size(self) -> int:
        """|V|: number of nodes."""
        return _node_count(self._root)

    @property
    def is_empty(self) -> bool:
        """True iff the tree contains no nodes."""
        return self._root is None

    def values(self) -> List[float]:
        """Return all node values in BFS order."""
        return bfs_values(self._root)

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify(self, verbose: bool = True) -> bool:
        """Verify all theorems for this tree.

        Parameters
        ----------
        verbose : bool
            Print individual theorem results if True.

        Returns
        -------
        bool
            True iff every theorem passes.
        """
        results: List[TheoremResult] = verify_all(self._root)
        all_pass = all(r.passed for r in results)
        if verbose:
            for r in results:
                status = "✓" if r.passed else "✗"
                print(f"  [{status}] {r}")
        return all_pass

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def print(self, label: str = "") -> None:
        """Print the tree in BFS-level order."""
        print_tree(self._root, label or f"ABT (h={self.height}, n={self.size})")

    def __repr__(self) -> str:
        return f"ABT(height={self.height}, size={self.size})"

    def __len__(self) -> int:
        return self.size
