# Average Binary Tree (ABT)

[![Tests](https://github.com/vikaskumar/average-binary-tree/actions/workflows/test.yml/badge.svg)](https://github.com/vikaskumar/average-binary-tree/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org)

Reference implementation of the **Average Binary Tree (ABT)** — a formally defined derived data structure arising from the sibling-averaging operator **f : T → T′** applied to a perfect binary tree.

> Vikas Kumar, *"Average Binary Tree (ABT): A Novel Formal Tree-to-Tree Transformation with Pairwise Sibling Aggregation"*, IJIRT Volume 13 Issue 1, ISSN 2349-6002, June 2026.

---

## Research Motivation

Tree-shaped data structures occupy a central role in theoretical computer science and practical algorithm design. Despite a long history, the question of how to *transform* one binary tree into another while preserving structural identity — rather than collapsing it into a scalar summary — has received comparatively little formal treatment.

The ABT addresses this gap: it takes a **perfect binary tree as input**, applies a **pairwise sibling-averaging rule**, and produces a new **binary tree as output** equipped with formal structural theorems. No prior work simultaneously satisfies all four criteria (Table I of the paper).

---

## Formal Definition

**Definition 6 (ABT).** Let (T, w) be a valued perfect binary tree of height h ≥ 1. The Average Binary Tree T′ = f(T, w) is constructed as:

- **(D1) Node set.** Bijection between V′ and the internal nodes of T.
- **(D2) Value mapping.** For each internal node p: `w′(f(p)) = (w(L(p)) + w(R(p))) / 2`
- **(D3) Edge preservation.** Parent-child topology of T's internal subgraph is preserved in T′.
- **(D4) Height.** `h(T′) = h(T) − 1`

---

## Theorems (Section IV)

| Result | Statement | Significance |
|--------|-----------|--------------|
| **Theorem 1** (Size) | `\|T′\| = (n−1)/2` | ≈50% node reduction |
| **Theorem 2** (Height) | `h(T′) = h(T)−1` | one-level compression |
| **Corollary 3** | `h(fᵏ(T)) = h(T)−k` | multi-scale hierarchy |
| **Theorem 4** (Bounds) | `min ≤ w′(u) ≤ max` | range-preserving |
| **Theorem 5** (Linearity) | `f(aT₁+bT₂) = af(T₁)+bf(T₂)` | supports fusion |
| **Property 1** | `\|V′\| < \|V\|` | strict compression |
| **Property 2** | isomorphism-preserving | topology stable |

---

## Quick Start

```bash
pip install average-binary-tree
```

```python
from abt import ABT

# Build from BFS-order list
tree = ABT.from_list([10, 6, 14, 4, 8, 12, 16])
tree.print()
# T (h=2, n=7):
#   L0: [10]
#   L1: [6, 14]
#   L2: [4, 8, 12, 16]

# Apply the operator
result = tree.transform()
result.print()
# T' (h=1, n=3):
#   L0: [10.0]
#   L1: [6.0, 14.0]

# Verify all theorems automatically
result.verify()
# ✓ [PASS] Theorem 1 (Size Contraction): expected=3 got=3
# ✓ [PASS] Theorem 2 (Height Reduction): expected=1 got=1
# ...

# Iterative contraction sequence (Corollary 3)
seq = tree.iterative_sequence()
for i, t in enumerate(seq):
    print(f"f^{i}(T): h={t.height}, n={t.size}")
```

### Extensions

```python
from abt import WABT, MBT, PBT
from abt import build_tree_from_list

T = build_tree_from_list([10, 6, 14, 4, 8, 12, 16])

# Weighted ABT (Section VII-A)
wabt = WABT(default_weights=(2.0, 1.0))
T_wabt = wabt.build(T)

# Median Binary Tree (Section VII-B)
T_mbt = MBT.build(T)

# Probabilistic Binary Tree (Section VII-C)
pbt = PBT(default_probs=(0.7, 0.3))
T_pbt = pbt.build(T)
```

### Theorem Verification

```python
from abt import build_tree_from_list
from abt.math import print_verification_report

T = build_tree_from_list([10, 6, 14, 4, 8, 12, 16])
print_verification_report(T, label="canonical h=2")
```

### Visualization

```python
from abt.viz import ascii_tree, plot_transformation, plot_interactive
from abt import build_tree_from_list, build_abt

T      = build_tree_from_list([10, 6, 14, 4, 8, 12, 16])
T_prime = build_abt(T)

print(ascii_tree(T, "Original"))        # ASCII
plot_transformation(T, T_prime)         # Matplotlib side-by-side
plot_interactive(T_prime)               # Plotly interactive
```

---

## Reproduce the Paper

```bash
python reproduce_paper.py
```

This reproduces all six test categories, all theorem verifications, Table III benchmarks, and extension demonstrations with one command.

---

## Repository Structure

```
average-binary-tree/
├── abt/                        # Core library
│   ├── core.py                 # Node, build_abt (Algorithm 1), helpers
│   ├── _facade.py              # ABT high-level class
│   ├── math/
│   │   ├── definitions.py      # Definitions 1-9 (executable)
│   │   └── theorems.py         # Theorems 1-5, Properties 1-2 (verifiable)
│   ├── extensions/
│   │   └── variants.py         # WABT, MBT, PBT (Section VII)
│   ├── viz/
│   │   └── visualizer.py       # ASCII, DOT, NetworkX, Matplotlib, Plotly
│   └── benchmarks/
│       └── runner.py           # Table III reproduction
├── tests/
│   ├── unit/
│   │   ├── test_core.py        # Six paper test categories + extras
│   │   ├── test_theorems.py    # Theorem verification tests
│   │   └── test_extensions.py  # WABT/MBT/PBT tests
│   └── conftest.py
├── reproduce_paper.py          # Single-command paper reproduction
├── pyproject.toml
├── CITATION.cff
└── README.md
```

---

## Algorithm (Section V)

**Algorithm 1 BuildABT(T, w)** — O(n) time, O(n) space BFS.

```
Require: Valued perfect binary tree (T, w), h(T) ≥ 1
1.  r′ ← Node((w(L(r)) + w(R(r))) / 2)
2.  Qorig ← {r};  Qnew ← {r′}
3.  while Qorig ≠ ∅:
4.      p ← Qorig.dequeue();  u ← Qnew.dequeue()
5.      if L(L(p)) ≠ ∅:       ▷ p's children are internal
6.          aL ← (w(L(L(p))) + w(R(L(p)))) / 2
7.          aR ← (w(L(R(p))) + w(R(R(p)))) / 2
8.          L(u) ← Node(aL);  R(u) ← Node(aR)
9.          Enqueue L(p), R(p) into Qorig
10.         Enqueue L(u), R(u) into Qnew
11. return r′
```

---

## Benchmarks (Section VI-C, Table III)

All three methods scale linearly. The ABT uniquely provides a **navigable tree** as output.

| Method | h=8 | h=10 | h=12 | h=14 |
|--------|-----|------|------|------|
| ABT (navigable tree) | fast | fast | fast | fast |
| STA (flat array) | faster | faster | faster | faster |
| APY (list of arrays) | fastest | fastest | fastest | fastest |

Run `python reproduce_paper.py` for exact numbers on your machine.

---

## Extensions (Section VII)

| Variant | Formula | Strength |
|---------|---------|---------|
| **ABT** | `(v₁+v₂)/2` | simplicity, general use |
| **WABT** | `(ω₁v₁+ω₂v₂)/(ω₁+ω₂)` | priority-aware (sensors, traffic) |
| **MBT** | `median(v₁,v₂)` | outlier-robust (noisy data) |
| **PBT** | `p₁v₁+p₂v₂` | uncertainty-aware (Bayesian) |

---

## Citation

```bibtex
@article{kumar2026abt,
  author  = {Kumar, Vikas},
  title   = {Average Binary Tree ({ABT}): A Novel Formal Tree-to-Tree
             Transformation with Pairwise Sibling Aggregation},
  journal = {International Journal of Innovative Research in Technology},
  volume  = {13},
  number  = {1},
  year    = {2026},
  month   = {June},
  issn    = {2349-6002},
  note    = {IJIRT 140001},
  url     = {https://ijirt.org/Article?manuscript=140001}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
