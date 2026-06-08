#!/usr/bin/env python3
"""
examples/quickstart.py
======================
Minimal working example of the ABT library.

Demonstrates the full workflow:
  1. Build a perfect binary tree
  2. Apply the ABT operator
  3. Verify all theorems
  4. Apply extensions (WABT, MBT, PBT)
  5. ASCII visualize the transformation
  6. Iterative contraction sequence

Run:
    python examples/quickstart.py
"""

from abt import ABT, WABT, MBT, PBT, build_tree_from_list, build_abt
from abt.viz import ascii_tree, side_by_side_ascii
from abt.math import print_verification_report

print("=" * 55)
print("  Average Binary Tree (ABT) — Quick Start")
print("=" * 55)

# ── 1. Build a height-2 perfect binary tree ───────────────────
print("\n1. Building perfect binary tree (h=2, n=7):")
values = [10, 6, 14, 4, 8, 12, 16]
tree = ABT.from_list(values)
tree.print("T (original)")

# ── 2. Apply the ABT operator ─────────────────────────────────
print("\n2. Applying the sibling-averaging operator f : T → T':")
result = tree.transform()
result.print("T' (ABT)")
print(f"\n   T  → h={tree.height}, n={tree.size}")
print(f"   T' → h={result.height}, n={result.size}  (≈50% node reduction)")

# ── 3. ASCII side-by-side ─────────────────────────────────────
print("\n3. Side-by-side comparison:")
print(side_by_side_ascii(tree.root, result.root))

# ── 4. Theorem verification ───────────────────────────────────
print("\n4. Automatic theorem verification:")
print_verification_report(tree.root, label="h=2 canonical")

# ── 5. Extensions ─────────────────────────────────────────────
print("\n5. Extensions (Section VII):")
T = build_tree_from_list(values)

wabt = WABT(default_weights=(2.0, 1.0))
T_wabt = wabt.build(T)
print(f"   WABT (ω₁=2, ω₂=1): {T_wabt.val:.4f}  [plain ABT: {result.root.val:.4f}]")

T_mbt = MBT.build(T)
print(f"   MBT  (median):       {T_mbt.val:.4f}")

pbt = PBT(default_probs=(0.7, 0.3))
T_pbt = pbt.build(T)
print(f"   PBT  (p₁=0.7):       {T_pbt.val:.4f}")

# ── 6. Iterative contraction (Corollary 3) ────────────────────
print("\n6. Iterative contraction sequence (Corollary 3):")
big_tree = ABT.from_list([
    100, 50, 150, 25, 75, 125, 175,
     10, 40,  60, 90, 110, 140, 160, 190
])
seq = big_tree.iterative_sequence()
for i, t in enumerate(seq):
    bar = "█" * t.size
    print(f"   f^{i}(T): h={t.height}, n={t.size:2d}  {bar}")

print("\nDone. See reproduce_paper.py for full paper reproduction.")
